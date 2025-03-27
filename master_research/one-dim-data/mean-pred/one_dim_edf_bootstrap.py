# 一次元データに対するBS法 図の描画など
import numpy as np
import matplotlib.pyplot as plt

# 乱数生成用のシードを設定
np.random.seed(42)
# 乱数生成の回数
ite = 5
# ランダムな整数値をシード値として取得
random_seed = np.random.randint(0, 10000, ite)  # 例えば 0 〜 9999 の間の整数をite個生成

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
    
n_samples = 1000 # サンプリング回数
    
for seed in np.nditer(random_seed):
    # 取得した乱数を新しいシード値として設定
    np.random.seed(seed)
    # データの生成
    data = np.random.randn(50)
    # 元データのヒストグラム
    plt.hist(data, bins=20, alpha=0.7, color='blue', label='Original Data')

    # ブートストラップサンプリングの実行
    bootstrap_samples = bootstrap(data, n_samples)

    # ブートストラップサンプルの平均を計算
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
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