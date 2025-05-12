import numpy as np
import matplotlib.pyplot as plt
import statistics


np.random.seed(42) # 乱数生成用のシードを設定
ite = 5 # 乱数生成の回数

# 2次元正規分布の平均ベクトルと共分散行列を設定
original_mean = [5, 5]  # 平均ベクトル
original_cov = [[1, 0.8], [0.8, 1]]  # 共分散行列（相関あり）

random_seed = np.random.randint(0, 10000, ite)  # ランダムな整数値をシード値として取得.例えば 0 〜 9999 の間の整数をite個生成
print("random_seed", random_seed) # 乱数シード値の確認

def bootstrap(data, num_dataset):
    """
    ブートストラップサンプリングを行う関数
    :param data: 元データ
    :param n_samples: サンプリング回数
    :return: ブートストラップサンプル
    """
    len_data = len(data)  # 元データのサイズ
    bootstrap_samples = []  # ブートストラップサンプルを格納する配列

    for i in range(num_dataset):
        indices = np.random.choice(len_data, size=len_data, replace=True)  # リサンプリングし、インデックスを取得
        bootstrap_samples.append(data[indices])   # サンプルを格納
    bootstrap_samples = np.array(bootstrap_samples)  # リストをNumPy配列に変換

    return bootstrap_samples


    
num_dataset = 1000 # サンプリング回数
dataset_size = 50 # 元データのサイズ 
    
for seed in np.nditer(random_seed):
    print("############################################")
    print("Random Seed:", seed) # 開始の合図
    print("############################################")
    np.random.seed(seed) # 取得した乱数を新しいシード値として設定
    data = np.random.multivariate_normal(original_mean, original_cov, size=dataset_size) # 学習元データの生成 (50, 2)
    print("data.shape", data.shape) # (50, 2)

    # 学習元データを可視化する
    # plt.hist(data, bins=20, alpha=0.7, color='blue', label='Original Data') # 元データのヒストグラム
    # plt.title("Original Data") # グラフの装飾
    # plt.xlabel("Value")
    # plt.ylabel("Frequency") 
    # plt.legend() # 凡例表示
    # plt.grid(True) # グリッド表示
    # plt.show() # 描画



    # ブートストラップ処理
    bootstrap_samples = bootstrap(data, num_dataset) # ブートストラップサンプリングの実行, n_samples個のデータセットを作成
    print("bootstrap_samples.shape", bootstrap_samples.shape) # (1000, 50, 2)
    # ---------------------統計量制御---------------------
    bootstrap_mean_vecs = np.mean(bootstrap_samples, axis=1) # 各ブートストラップサンプルの平均ベクトル（1000, 2）
    bootstrap_cov_mats = np.array([np.cov(sample.T) for sample in bootstrap_samples]) # 各ブートストラップサンプルの共分散行列（1000, 2, 2）
    bootstrap_corr_coefs = np.array([np.corrcoef(sample.T)[0, 1] for sample in bootstrap_samples]) # 各ブートストラップサンプルの相関係数（1000,）

    # サイズの確認
    # print("Bootstrap Mean Vector", bootstrap_mean_vecs.shape) # 平均ベクトル
    # print("Bootstrap Covariance Matrix", bootstrap_cov_mats.shape) # 共分散行列
    # print("Bootstrap Correlation Coefficient", bootstrap_corr_coefs.shape) # 相関係数

    # 代表値を出力
    print("平均ベクトルの平均:", np.mean(bootstrap_mean_vecs, axis=0))
    print("共分散行列の平均:\n", np.mean(bootstrap_cov_mats, axis=0))
    print("相関係数の平均:", np.mean(bootstrap_corr_coefs))





    # 平均ベクトルの分布
    
    plt.figure(figsize=(6, 6))
    plt.scatter(bootstrap_mean_vecs[:, 0], bootstrap_mean_vecs[:, 1], alpha=0.5, color='teal', label='Bootstrap Means')

    # 原点を基準として赤十字
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.scatter(0, 0, color='red', marker='x', s=100, label='True Mean')

    plt.xlabel('Mean of X1')
    plt.ylabel('Mean of X2')
    plt.title('Scatter Plot of Bootstrap Mean Vectors')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()



    from scipy.stats import gaussian_kde
    from mpl_toolkits.mplot3d import Axes3D  # 必要

    # カーネル密度推定
    kde = gaussian_kde(bootstrap_mean_vecs.T)

    # グリッド生成
    x = np.linspace(np.min(bootstrap_mean_vecs[:, 0]) - 0.5, np.max(bootstrap_mean_vecs[:, 0]) + 0.5, 100)
    y = np.linspace(np.min(bootstrap_mean_vecs[:, 1]) - 0.5, np.max(bootstrap_mean_vecs[:, 1]) + 0.5, 100)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    # 3Dプロット
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

    ax.set_xlabel('Mean of X1')
    ax.set_ylabel('Mean of X2')
    ax.set_zlabel('Density')
    ax.set_title('Bird\'s Eye View of Bootstrap Mean Vectors (KDE)')
    plt.tight_layout()
    plt.show()




    # 相関係数の分布
    plt.figure(figsize=(6, 4))
    plt.hist(bootstrap_corr_coefs, bins=30, color='orchid')
    plt.title('Bootstrap Correlation Coefficients')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()




    # 終了の合図
    print("############################################")
    print("End")
    print("############################################")

    # 改行
    print("\n")