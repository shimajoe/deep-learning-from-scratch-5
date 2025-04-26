from sklearn.preprocessing import StandardScaler
import numpy as np

# 例：2次元正規分布データ（50個）
data = np.random.multivariate_normal([5, 5], [[1, 0.3], [0.3, 1]], size=50)

# データの平均・分散を確認
print("元の平均:", data.mean(axis=0))     # → [5, 5]
print("元の標準偏差:", data.std(axis=0))  # → [1, 1]

# スケーラーを生成・適合（fit）し、変換（transform）
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 学習後の平均・分散を確認
print("平均:", scaled_data.mean(axis=0))     # → ほぼ [0, 0]
print("標準偏差:", scaled_data.std(axis=0))  # → ほぼ [1, 1]


# スケーリングをもとに戻す
original_data = scaler.inverse_transform(scaled_data)
print("元のデータに戻す:", original_data.mean(axis=0))  # → ほぼ [5, 5]
print("元のデータに戻す:", original_data.std(axis=0))   # → ほぼ [1, 1]
