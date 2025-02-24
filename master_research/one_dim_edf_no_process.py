# BSによるデータセット1つに対する描画など
from matplotlib import pyplot as plt
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# 学習データ(ガウスノイズ)
data = np.random.randn(50)  # shape: (50,)
scaler = StandardScaler()
data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
print("Train Data:", data)

# PyTorchテンソルに変換
train_data = torch.tensor(data, dtype=torch.float32).view(-1, 1)

# 必要ならCPUに移動
train_data = train_data.cpu()

# NumPy配列に変換
train_data_np = train_data.numpy()

# 重複ありのサンプリング
sampled_data = np.random.choice(train_data_np.flatten(), size=50, replace=True)

print("元データ:", train_data_np.flatten())
print("サンプリングデータ:", sampled_data)

# 元データのヒストグラム
plt.hist(train_data_np.flatten(), bins=20, alpha=0.6, color='blue', label='Original Data')

# サンプリングデータのヒストグラム
plt.hist(sampled_data, bins=20, alpha=0.6, color='orange', label='Sampled Data')

# 共通のビン境界を定義
min_value = min(train_data_np.min(), sampled_data.cpu().numpy().min())  # 最小値
max_value = max(train_data_np.max(), sampled_data.cpu().numpy().max())  # 最大値
bins = np.linspace(min_value, max_value, 21)  # 20個の区間に分ける（ビン境界を21個）

# グラフの装飾
plt.title('Histogram Comparison: Original vs Sampled Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.grid(True)

# 表示
plt.show()