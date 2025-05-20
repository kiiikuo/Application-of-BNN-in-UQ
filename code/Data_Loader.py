import netCDF4 as nc
import numpy as np
import torch
import random

def load_data(n, num_samples=50, seed=42):
    """
    从指定数据文件中，随机选择多个滑动窗口作为训练样本。
    每个窗口使用 m, m+1, m+2 作为输入，m+3 作为目标输出。

    返回：
        inputs: Tensor, shape (num_samples, 3, 320, 416)
        targets: Tensor, shape (num_samples, 320, 416)
    """
    random.seed(seed)

    file_path = f"E:\\TAIR Data\\Combined_TAIR_{n}.nc"
    print(f"加载文件: {file_path}")
    ds = nc.Dataset(file_path)
    TAIR = ds.variables['TAIR'][:]

    if TAIR.shape != (200, 320, 416):
        TAIR = np.transpose(TAIR, (2, 1, 0))  # shape: (T, H, W)

    max_start = 196  # 允许的最大起点索引（防止越界）
    start_indices = random.sample(range(max_start + 1), num_samples)

    input_list = []
    target_list = []

    for m in start_indices:
        input_frames = []
        for offset in range(3):  # m, m+1, m+2
            frame = TAIR[m + offset]
            frame = np.where(frame == -999, np.nan, frame)
            input_frames.append(torch.tensor(frame, dtype=torch.float32))
        input_tensor = torch.stack(input_frames)  # (3, 320, 416)
        input_list.append(input_tensor)

        target_frame = TAIR[m + 3]
        target_frame = np.where(target_frame == -999, np.nan, target_frame)
        target_tensor = torch.tensor(target_frame, dtype=torch.float32)
        target_list.append(target_tensor)

    inputs = torch.stack(input_list)    # shape: (num_samples, 3, 320, 416)
    targets = torch.stack(target_list)  # shape: (num_samples, 320, 416)

    print(f"生成训练样本数: {inputs.shape[0]}")
    return inputs, targets
