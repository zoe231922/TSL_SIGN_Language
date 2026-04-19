import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import glob
import json

# --- 1. LSTM 模型定義 (126 維手部特徵版本) ---
class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=126, hidden_size=128, num_layers=2, num_classes=10):
        super(SignLanguageLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 層 (支持 dropout 當層數 > 1)
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, 
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # BatchNormalization (在全連接層前)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
        # 全連接層
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # 初始化隱藏狀態
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM 前向傳播
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 取最後一個時序的輸出
        last_output = lstm_out[:, -1, :]
        
        # BatchNormalization
        bn_out = self.batch_norm(last_output)
        
        # Dropout
        dropout_out = self.dropout(bn_out)
        
        # 全連接層
        logits = self.fc(dropout_out)
        
        return logits

# --- 2. 數據集載入器 ---
class SignDataset(Dataset):
    def __init__(self, dataset_path, seq_len=30, input_size=126):
        self.seq_len = seq_len
        self.input_size = input_size
        self.data = []
        self.labels = []
        
        # 檢查資料集目錄是否存在
        if not os.path.exists(dataset_path):
            print(f"❌ 錯誤: 找不到目錄 '{dataset_path}'")
            print(f"\n請確保：")
            print(f"  1. 您已在專案根目錄運行此腳本")
            print(f"  2. 已執行過 '自動化採集工具.py' 採集訓練數據")
            print(f"  3. 已執行過 'process_data.py' 進行數據預處理")
            print(f"\n建議流程：")
            print(f"  $ python 自動化採集工具.py   # 採集手語詞彙")
            print(f"  $ python process_data.py        # 預處理數據")
            print(f"  $ python train_model.py         # 訓練模型")
            raise FileNotFoundError(f"資料集目錄不存在: {dataset_path}")
        
        self.classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        
        if len(self.classes) == 0:
            print(f"❌ 錯誤: '{dataset_path}' 目錄內找不到任何手語詞彙子目錄")
            print(f"\n'{dataset_path}' 應包含如下結構：")
            print(f"  sign_dataset/")
            print(f"    ├── 一/       (手語詞彙：數字「一」)")
            print(f"    ├── 二/       (手語詞彙：數字「二」)")
            print(f"    └── ...")
            raise ValueError(f"未找到任何手語詞彙類別")
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(dataset_path, cls_name)
            # 依據需求改為載入 .npy 格式 (立體結構：Sample, Frame, Feature)
            npy_files = glob.glob(os.path.join(cls_dir, "*.npy"))
            
            # 只保留由 process_data.py 產生的分段檔案 (含 _seg 的檔案)
            # 排除舊版整段錄製的 .npy（shape 為 30×345，格式不同）
            npy_files = [f for f in npy_files if '_seg' in os.path.basename(f)]
            
            if len(npy_files) == 0:
                print(f"⚠️  警告: '{cls_name}' 目錄內找不到 .npy 檔案")
                print(f"   請確保已執行 'process_data.py' 進行數據預處理")
                continue
            
            for f in npy_files:
                features = np.load(f)  # 形狀應為 (30, 126)
                
                # 驗證特徵維度 (排除舊版 345 維資料)
                if features.ndim != 2 or features.shape[1] != self.input_size:
                    print(f"   ⚠️  跳過不相容檔案: {os.path.basename(f)} (shape: {features.shape})")
                    continue
                
                # 再次確認時間序列長度 (防呆)
                if len(features) != self.seq_len:
                    if len(features) >= self.seq_len:
                        indices = np.linspace(0, len(features) - 1, self.seq_len).astype(int)
                        features = features[indices]
                    else:
                        pad_size = self.seq_len - len(features)
                        padding = np.zeros((pad_size, features.shape[1]))
                        features = np.vstack([features, padding])
                
                self.data.append(features)
                self.labels.append(self.class_to_idx[cls_name])
                
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.longlong)
        print(f"數據集載入完成: 共有 {len(self.data)} 筆資料, 標籤類別: {self.classes}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])

# --- 3. 訓練主程式 ---
def train():
    # 參數設定
    dataset_path = "sign_dataset"
    seq_len = 30
    input_size = 126 # 42 點 * 3 軸
    hidden_size = 128
    num_layers = 2
    batch_size = 16
    num_epochs = 100
    learning_rate = 0.001
    
    # 檢查工作目錄
    print(f"📁 當前工作目錄: {os.getcwd()}")
    print(f"📁 尋找資料集路徑: {os.path.abspath(dataset_path)}\n")
    
    # 準備數據
    try:
        dataset = SignDataset(dataset_path, seq_len, input_size)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ 資料集加載失敗")
        print(f"   錯誤原因: {e}")
        return
        
    if len(dataset) == 0:
        print("❌ 錯誤: 找不到任何訓練資料")
        print(f"\n請確保已執行以下步驟：")
        print(f"  1. python 自動化採集工具.py   # 採集手語詞彙")
        print(f"  2. python process_data.py        # 預處理並產生 .npy 檔案")
        return
    print(f"✅ 數據集加載成功")
    print(f"   - 訓練樣本數: {len(dataset)}")
    print(f"   - 手語詞彙類別: {', '.join(dataset.classes)}\n")
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_classes = len(dataset.classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignLanguageLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 開始訓練
    print(f"🚀 開始在 {device} 上訓練 LSTM 模型...")
    print(f"   - 輪數: {num_epochs}")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - 學習率: {learning_rate}\n")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (features, labels) in enumerate(dataloader):
            features = features.to(device)
            labels = labels.to(device)
            
            # 前向傳播
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
            
    # 儲存模型與配置
    torch.save(model.state_dict(), 'sign_model.pth')
    config = {
        'classes': dataset.classes,
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'seq_len': seq_len,
        'description': f'{input_size}-dimensional features: Hands only (42 points x 3 axes)'
    }
    with open('model_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ 模型訓練完成！")
    print(f"   - 模型參數已儲存: sign_model.pth")
    print(f"   - 配置檔案已儲存: model_config.json")
    print(f"   - 可識別詞彙: {', '.join(dataset.classes)}")
    print(f"\n🎯 下一步：執行即時辨識")
    print(f"   $ python realtime_recognition.py")

if __name__ == "__main__":
    train()
