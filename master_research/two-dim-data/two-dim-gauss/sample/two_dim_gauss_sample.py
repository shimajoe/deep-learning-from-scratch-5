import numpy as np

# 乱数生成用のシードを設定
np.random.seed(42)
# 例：2次元正規分布から1000サンプル生成
mean = [0, 0]  # 平均ベクトル
cov = [[1, 0.8], [0.8, 1]]  # 共分散行列（相関あり）

data = np.random.multivariate_normal(mean, cov, size=50) 
print("data", data)
print("data.shape", data.shape) # (50, 2)

# 平均ベクトル
mean_vec = np.mean(data, axis=0)  # => shape (2,)
print("mean_vec", mean_vec)
# 共分散行列
cov_mat = np.cov(data.T)  # => shape (2, 2)
print("cov_mat", cov_mat)
# 相関係数行列
corr_mat = np.corrcoef(data.T)  # => shape (2, 2)
corr_coef = corr_mat[0, 1]  # 相関係数ρ scalar
print("corr_mat", corr_mat)
print("corr_coef", corr_coef) # 相関係数ρ

print("type of data", type(data)) # ndarray
print("data_length", len(data)) # データサイズ 50


# 可視化
import matplotlib.pyplot as plt

plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('2D Gaussian Samples')
plt.axis('equal')
plt.show()