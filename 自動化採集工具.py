import cv2
import mediapipe as mp
import time
import os
import csv

def record_sign_language_expert():
    # 1. 初始化 MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.5
    )

    # 2. 設定標籤與路徑
    label_name = input("請輸入手語單字名稱 (例如: hello): ").strip()
    save_dir = os.path.join("sign_dataset", label_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"建立資料夾: {save_dir}")

    # 3. 設定攝影機與影片參數
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    recording = False
    video_out = None
    csv_file = None
    csv_writer = None

    print(f"\n--- 準備錄製單字：【{label_name}】 ---")
    print("操作說明:")
    print("  [SPACE] - 開始或停止錄影")
    print("  [ESC]   - 退出程式")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 影像翻轉與顏色轉換
        frame = cv2.flip(frame, 1) # 鏡像翻轉，自己看比較自然
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        display_frame = frame.copy()
        current_landmarks = []

        # 提取座標並繪製骨架
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 取得 21 個點的 (x, y, z)
                for lm in hand_landmarks.landmark:
                    current_landmarks.extend([lm.x, lm.y, lm.z])
        
        # 補齊 126 個數值 (確保雙手 42 點 * 3 軸資料長度固定)
        while len(current_landmarks) < 126:
            current_landmarks.append(0.0)

        # UI 狀態顯示
        status = "RECORDING" if recording else "READY"
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(display_frame, f"{status}: {label_name}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Sign Language Data Collector', display_frame)

        # 鍵盤監聽
        key = cv2.waitKey(1) & 0xFF
        
        # 空白鍵：開始/停止
        if key == ord(' '):
            if not recording:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                video_path = os.path.join(save_dir, f"{label_name}_{timestamp}.mp4")
                csv_path = os.path.join(save_dir, f"{label_name}_{timestamp}.csv")
                
                # 初始化寫入器
                video_out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                csv_file = open(csv_path, 'w', newline='')
                csv_writer = csv.writer(csv_file)
                
                # 寫入 CSV 標題 (pt0_x, pt0_y, pt0_z...)
                header = [f"pt{i}_{axis}" for i in range(42) for axis in ['x', 'y', 'z']]
                csv_writer.writerow(header)
                
                recording = True
                print(f"● 錄製中: {video_path}")
            else:
                recording = False
                if video_out: video_out.release()
                if csv_file: csv_file.close()
                print("■ 錄製結束並儲存數據。")

        # 錄製中寫入數據
        if recording:
            video_out.write(frame)
            csv_writer.writerow(current_landmarks)

        # ESC 退出
        if key == 27:
            break

    # 釋放資源
    cap.release()
    if video_out: video_out.release()
    if csv_file: csv_file.close()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    record_sign_language_expert()