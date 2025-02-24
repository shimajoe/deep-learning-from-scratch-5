# 一次元データに対するBS法 図の描画など
import numpy as np
import matplotlib.pyplot as plt

# 元データの作成
np.random.seed(42)
data = np.random.randn(50)

# ブートストラップサンプリング関数の定義
def bootstrap(data, n_samples):
    """
    :param data: 元データ(一次元配列)
    :param n_samples: サンプリング回数
    :return: ブートストラップされたサンプル群
    """
    n = len(data)
    bootstrap_samples = []
    for _ in range(n_samples):
        sample = np.random.choice(data, n, replace=True)
        bootstrap_samples.append(sample)
        return np.array(bootstrap_samples)
    
# ブートストラップサンプリングの実行
n_samples = 1000 # サンプリング回数
bootstrap_samples = bootstrap(data, n_samples)

# ブートストラップサンプルの平均を計算
bootstrap_means = np.mean(bootstrap_samples, axis=1)

# 元データのヒストグラム
plt.hist(data, bins=20, alpha=0.7, color='blue', label='Original Data')

# ブートストラップサンプル平均のヒストグラム
plt.hist(bootstrap_means, bins=20, alpha=0.7, color='orange', label='Bootstrap Sample Means')

# グラフの装飾
plt.title("Bootstrap Sampling and Original Data Comparison")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

# 表示
plt.show()