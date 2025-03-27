# 経験分布関数からのBS法
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# ハイパーパラメータ
num_timesteps = 1000 # 拡散ステップ数
epochs = 10          # 学習エポック数
lr = 1e-3            # 学習率
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Pytorchの組み込み演算により効率的に計算
# 高次元データ用のtorch.arrangeやtorch.expを活用してコードを簡潔に

# 時間埋め込み（正弦波位置エンコーディング）
def pos_encoding(timesteps, output_dim, device='cpu'):
    position = timesteps.view(-1, 1).float()  # 必要に応じて型変換
    div_term = torch.exp(torch.arange(0, output_dim, 2, device=device, dtype=torch.float32) * 
                         (-np.log(10000.0) / output_dim))
    sinusoid = torch.cat([torch.sin(position * div_term), torch.cos(position * div_term)], dim=1)
    return sinusoid

# Dropoutの導入: 過学習を防ぐために、各隠れ層にnn.Dropoutを追加。
# Batch Normalizationの導入: 学習を安定させるためにnn.BatchNorm1dを適用。
# 活性化関数の選択: F.reluの代わりにnn.LeakyReLUやnn.ELUを試すことで、勾配消失問題に対応。

# 拡散モデル
class DiffusionModel(nn.Module):
    def __init__(self, time_embed_dim=16):
        super(DiffusionModel, self).__init__()
        self.time_embed_dim = time_embed_dim  # time_embed_dimをインスタンス変数として初期化
        self.fc1 = nn.Linear(1 + time_embed_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, t):
        # 時間埋め込み
        t_embed = pos_encoding(t, self.time_embed_dim, x.device)
        x_t = torch.cat([x, t_embed], dim=1)  # 時間情報と入力データを結合
        x_t = F.relu(self.fc1(x_t))
        x_t = F.relu(self.fc2(x_t))
        return self.fc3(x_t)

# 拡散プロセス
class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        t_idx = t - 1 # alphas[0] is for t=1
        alpha_bar = self.alpha_bars[t_idx].view(-1, 1)  # (N, 1)
        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()
        
        t_idx = t - 1 # alphas[0] is for t=1
        alpha = self.alphas[t_idx].view(-1, 1)
        alpha_bar = self.alpha_bars[t_idx].view(-1, 1)
        model.eval()
        with torch.no_grad():
            eps = model(x, t)

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0  # no noise at t=1

        mu = (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha)

        return mu

# モデルとオプティマイザ
time_embed_dim = 16
model = DiffusionModel(time_embed_dim=time_embed_dim).to(device)
optimizer = Adam(model.parameters(), lr=lr)
diffuser = Diffuser(num_timesteps=num_timesteps, device=device)

# シード値の固定
np.random.seed(42)

# 学習データ(ガウスノイズ)
data = np.random.randn(50)  # shape: (50,)
scaler = StandardScaler()
data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
print("Train Data:", data)
train_data = torch.tensor(data, dtype=torch.float32).view(-1, 1).to(device)  # shape: (10, 1)

# データローダー作成
batch_size = 10
dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 学習ループ
losses = []
for epoch in range(epochs):
    loss_sum = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        x = batch.to(device)
        t = torch.randint(1, num_timesteps + 1, (len(x),), device=device)

        x_noisy, noise = diffuser.add_noise(x, t)
        noise_pred = model(x_noisy, t)
        loss = F.mse_loss(noise_pred, noise)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
    avg_loss = loss_sum / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

# 学習曲線のプロット
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# サンプリング
model.eval()
with torch.no_grad():
    B = 1000 # 群の数
    n = 50 # 1群あたりのサンプル数
    samples = torch.randn((n*B, 1), device=device)  # ランダムなノイズから開始
    for t in range(num_timesteps, 0, -1):
        t_tensor = torch.tensor([t] * len(samples), device=device)
        samples = diffuser.denoise(model, samples, t_tensor)
    samples = samples.view(n, B)

# 各群の平均値を計算
samples_group_means = samples.mean(dim=0)

# 結果を確認
print("Group Means Shape:", samples_group_means.shape)  # Output: torch.Size([1000])
print("Group Means:", samples_group_means.cpu().numpy())  # 必要ならNumPy配列に変換
