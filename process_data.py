import pandas as pd
import numpy as np
import os
import glob

def process_csv(file_path):
    # 自動偵測 CSV 是否有標題行
    # 舊版工具錄製的 CSV 無標題行（345欄：Pose+Face+Hands），新版才有 pt0_x...pt41_z 標題（126欄：Hands only）
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    
    first_cell = first_line.split(',')[0]
    try:
        float(first_cell)  # 若第一格是數字，表示無標題行（舊版格式）
        header = None
    except ValueError:
        header = 0  # 有標題行（新版格式）
    
    df = pd.read_csv(file_path, header=header)
    num_cols = len(df.columns)
    
    if header is None:
        # 舊版格式：345 欄（Pose:99 + Face:120 + Hands:126）
        # 只取最後 126 欄（手部資料）以和新版統一
        if num_cols == 345:
            df = df.iloc[:, 219:]  # 跳過 Pose(99) + Face(120) = 219 欄
        col_names = [f"pt{i}_{axis}" for i in range(42) for axis in ['x', 'y', 'z']]
        df.columns = col_names[:len(df.columns)]
    

    # 1. 偵測手部是否有出現 (所有值不為 0.0)
    # 我們檢查每一列的總和或是否有非零值
    # 只要 pt0_x 到 pt20_z 有任何一個不是 0，就代表至少偵測到一隻手
    has_hand = (df.iloc[:, :126] != 0).any(axis=1)
    
    # 2. 段落切分邏輯 (優化版)
    # 目的：填補短暫的消失 (Flicker)，並過濾掉太短的碎片
    
    hand_signal = has_hand.astype(int).tolist()
    
    # A. 填補短暫的中斷 (例如：1 0 0 1 -> 1 1 1 1)
    max_gap = 15 # 容忍最多 15 幀的中斷 (約 0.5 秒)
    for i in range(1, len(hand_signal)):
        if hand_signal[i-1] == 1 and hand_signal[i] == 0:
            # 往後找最近的 1
            for j in range(i + 1, min(i + max_gap, len(hand_signal))):
                if hand_signal[j] == 1:
                    # 找到 1 了，填補中間的 0
                    for k in range(i, j):
                        hand_signal[k] = 1
                    break
    
    # B. 找出所有連續的 1 段落
    segments = []
    if not hand_signal:
        return
        
    start_idx = -1
    for i, val in enumerate(hand_signal):
        if val == 1 and start_idx == -1:
            start_idx = i
        elif val == 0 and start_idx != -1:
            segments.append(list(range(start_idx, i)))
            start_idx = -1
    if start_idx != -1:
        segments.append(list(range(start_idx, len(hand_signal))))
        
    # C. 過濾掉太短的錄製段落 (例如：低於 15 幀的可能是誤報)
    min_seg_len = 15
    segments = [seg for seg in segments if len(seg) >= min_seg_len]

    print(f"檔案: {os.path.basename(file_path)}, 偵測到 {len(segments)} 個段落")


    # 3. 處理每個段落並儲存 (依據需求改為 .npy 格式，實現立體維度)
    base_name = os.path.splitext(file_path)[0]
    seq_len = 30 # 固定時間序列長度
    
    for idx, seg_indices in enumerate(segments):
        seg_df = df.iloc[seg_indices].copy()
        
        # 4. 手腕中心化 (Wrist Centering)
        for i in range(2): # 兩隻手
            wrist_x_col = f"pt{i*21}_x"
            wrist_y_col = f"pt{i*21}_y"
            wrist_z_col = f"pt{i*21}_z"
            
            wrists_x = seg_df[wrist_x_col].values.copy()
            wrists_y = seg_df[wrist_y_col].values.copy()
            wrists_z = seg_df[wrist_z_col].values.copy()
            
            mask = wrists_x != 0
            if mask.any():
                for j in range(i*21, (i+1)*21):
                    seg_df.loc[mask, f"pt{j}_x"] -= wrists_x[mask]
                    seg_df.loc[mask, f"pt{j}_y"] -= wrists_y[mask]
                    seg_df.loc[mask, f"pt{j}_z"] -= wrists_z[mask]
        
        # 5. 轉換為 NumPy 並進行時間序列正規化 (Resampling to 30 frames)
        # 提取特徵 (pt0_x ~ pt41_z)
        features = seg_df.iloc[:, :126].values # (len, 126)
        
        if len(features) >= seq_len:
            # 均勻抽樣
            indices = np.linspace(0, len(features) - 1, seq_len).astype(int)
            features = features[indices]
        else:
            # 補 0 (Padding)
            pad_size = seq_len - len(features)
            padding = np.zeros((pad_size, features.shape[1]))
            features = np.vstack([features, padding])
            
        # 6. 儲存為 .npy 檔案
        # 此時 features 的形狀為 (30, 126)，即為「立體」結構中的單一樣本
        output_path = f"{base_name}_seg{idx+1}.npy"
        np.save(output_path, features)
        print(f"  儲存 NumPy 序列 {idx+1} -> {output_path} (Shape: {features.shape})")

def main():
    dataset_path = "sign_dataset"
    label_dirs = glob.glob(os.path.join(dataset_path, "*"))
    
    for label_dir in label_dirs:
        if not os.path.isdir(label_dir):
            continue
        print(f"\n正在處理目錄: {label_dir}")
        csv_files = glob.glob(os.path.join(label_dir, "*.csv"))
        # 過濾掉已經切分過的檔案 (包含 _seg 的)
        csv_files = [f for f in csv_files if "_seg" not in f]
        
        for csv_file in csv_files:
            process_csv(csv_file)

if __name__ == "__main__":
    main()
