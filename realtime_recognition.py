import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import json
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# --- 1. 定義輔助函數：在 OpenCV 影像上繪製中文字 ---
def draw_chinese_text(img, text, position, font_size=25, color=(255, 255, 255)):
    # 將 OpenCV (BGR) 轉換為 PIL (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 載入 Windows 系統內建字體 (微軟雅黑)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
    except:
        font = ImageFont.load_default()
        
    draw.text(position, text, font=font, fill=color)
    
    # 將 PIL 轉回 OpenCV (BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- 2. 定義 LSTM 模型 (需與訓練時完全一致) ---
class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SignLanguageLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        # BatchNormalization (與 train_model.py 一致)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        # Dropout
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        last_output = out[:, -1, :]
        bn_out = self.batch_norm(last_output)
        dropout_out = self.dropout(bn_out)
        return self.fc(dropout_out)


def run_realtime():
    # 檢查模型檔案
    print(f"📁 當前工作目錄: {os.getcwd()}\n")
    
    if not os.path.exists('sign_model.pth'):
        print(f"❌ 錯誤: 找不到模型檔案 'sign_model.pth'")
        print(f"\n請確保您已完成以下步驟：")
        print(f"  1. python 自動化採集工具.py   # 採集手語詞彙")
        print(f"  2. python process_data.py        # 預處理數據")
        print(f"  3. python train_model.py         # 訓練模型 (生成 sign_model.pth)")
        print(f"\n💡 提示：")
        print(f"  - 訓練模型需要約 100 個 epoch")
        print(f"  - 若使用 GPU 會明顯加快訓練速度")
        print(f"  - 訓練完成後會自動生成 sign_model.pth")
        return
    
    if not os.path.exists('model_config.json'):
        print(f"❌ 錯誤: 找不到配置檔案 'model_config.json'")
        print(f"\n請重新執行訓練：")
        print(f"  $ python train_model.py")
        return

    print(f"✅ 模型檔案已找到")
    print(f"   - sign_model.pth")
    print(f"   - model_config.json\n")
    
    # 載入模型配置
    try:
        with open('model_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ 錯誤: 無法讀取配置檔案")
        print(f"   {str(e)}")
        return
    
    classes = config['classes']
    input_size = config['input_size']
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    seq_len = config['seq_len']
    
    print(f"🎯 可識別的手語詞彙: {', '.join(classes)}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用設備: {device}\n")
    
    try:
        model = SignLanguageLSTM(input_size, hidden_size, num_layers, len(classes)).to(device)
        model.load_state_dict(torch.load('sign_model.pth', map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ 錯誤: 無法加載模型")
        print(f"   {str(e)}")
        return
    
    print(f"✅ 模型加載成功\n")
    
    # MediaPipe 設定
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(f"❌ 錯誤: 無法打開攝影機")
        print(f"\n請檢查：")
        print(f"  1. 攝影機是否已連接")
        print(f"  2. 系統是否授予訪問權限")
        print(f"  3. 嘗試重新插入 USB 攝影機")
        return
    print(f"📹 攝影機已連接\n")
    
    # 建立一個序列緩衝區
    sequence_buffer = deque(maxlen=seq_len)
    confidence_threshold = 0.8
    
    # --- [新增] 延遲判定機制變數 (依據 readme3.md) ---
    CONFIRM_FRAMES = 10       # 延遲門檻：必須連續出現 10 幀才算判定成功
    consecutive_count = 0     # 連續出現次數計數器
    last_prediction_idx = -1  # 上一次預測的標籤索引
    final_confirmed_label = "等待中..." # 最終確認顯示的標籤
    current_confidence = 0.0

    print(f"🚀 即時辨識啟動！")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📋 使用說明：")
    print(f"  1. 將手放入攝影機視野")
    print(f"  2. 比出手語動作（保持約 1 秒）")
    print(f"  3. 系統會在穩定度達 100% 時辨識")
    print(f"  4. 按 ESC 退出程式")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        current_landmarks = []
        
        if results.multi_hand_landmarks:
            # 為了穩定性，我們處理每一隻手並做 Wrist Centering
            temp_all_hands = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 特徵提取
                lm_list = []
                for lm in hand_landmarks.landmark:
                    lm_list.append([lm.x, lm.y, lm.z])
                
                # 手腕中心化 (Wrist Centering)
                wrist = lm_list[0] # 第 0 個點是手腕
                centered_lm = []
                for pt in lm_list:
                    centered_lm.extend([pt[0] - wrist[0], pt[1] - wrist[1], pt[2] - wrist[2]])
                
                temp_all_hands.append(centered_lm)
            
            # 將手部數據合併為 126 維 (與訓練一致)
            for i in range(2): # 支援雙手
                if i < len(temp_all_hands):
                    current_landmarks.extend(temp_all_hands[i])
                else:
                    current_landmarks.extend([0.0] * 63)
            
            # 正常加入緩衝區
            sequence_buffer.append(current_landmarks)
        else:
            # --- [新增] 異常處理 (依據 readme3.md)：沒偵測到手時重置計數器與緩衝區 ---
            current_landmarks = [0.0] * 126
            consecutive_count = 0
            sequence_buffer.clear()
            final_confirmed_label = "等待中..."
            
        # 當緩衝區滿了，進行預測 (依據 readme2.md 點 3：使用 deque 並轉換為 NumPy 陣列)
        if len(sequence_buffer) == seq_len:
            # 轉換為 NumPy 陣列，形狀為 (seq_len, input_size) -> (30, 126)
            data_array = np.array(sequence_buffer)
            # 增加 Batch 維度，確保形狀為 (1, 30, 126) 符合 LSTM 輸入 (Batch, Seq, Feature)
            input_data = np.expand_dims(data_array, axis=0)
            
            # 轉換為 Tensor 並進行推論
            input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                max_prob, predicted_idx = torch.max(probs, dim=1)
                
                # --- [核心] 延遲判定邏輯處理 (依據 readme3.md & 範例.py) ---
                prob_val = max_prob.item()
                idx_val = predicted_idx.item()
                
                if prob_val > confidence_threshold:
                    # 如果當前預測與上一次相同，增加計數
                    if idx_val == last_prediction_idx:
                        consecutive_count = min(CONFIRM_FRAMES, consecutive_count + 1)
                    else:
                        # 預測改變，重置計數
                        consecutive_count = 1
                        last_prediction_idx = idx_val
                    
                    # 只有當連續次數達到門檻時，才更新最終標籤
                    if consecutive_count >= CONFIRM_FRAMES:
                        final_confirmed_label = classes[idx_val]
                        current_confidence = prob_val
                else:
                    # 信心值不足，緩慢減少計數增加穩定感
                    consecutive_count = max(0, consecutive_count - 1)

        # 顯示結果 UI
        # 計算穩定度百分比 (依據 readme3.md)
        stability_progress = int((consecutive_count / CONFIRM_FRAMES) * 100)
        color = (0, 255, 0) if consecutive_count >= CONFIRM_FRAMES else (0, 255, 255)
        
        # 繪製黑底半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # 使用 Pillow 繪製中文字 (顯示最終確認結果與穩定度)
        frame = draw_chinese_text(frame, f"辨識結果: {final_confirmed_label}", (10, 10), font_size=24, color=(255, 255, 255))
        frame = draw_chinese_text(frame, f"穩定度: {stability_progress}% ({current_confidence:.2f})", (10, 45), font_size=18, color=color)

        cv2.imshow('Sign Language Realtime Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print(f"\n✅ 辨識程式已關閉")

if __name__ == "__main__":
    print(f"")
    print(f"🤟 手語辨識系統 - 實時推論模式")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    run_realtime()
