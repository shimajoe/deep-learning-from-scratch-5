import torch
import time
def check_cuda():
    # CUDAが利用可能かどうかを確認
    if torch.cuda.is_available():
        print("CUDA is available! Using GPU...")
        device = torch.device("cuda")  # GPUを選択
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")  # 使用するGPUの名前を表示
    else:
        print("CUDA is not available. Using CPU...")
        device = torch.device("cpu")  # GPUが使用できない場合はCPUを使用
    size = 20000
    # サンプルテンソルを作成し、GPU/CPU上に転送
    x = torch.randn(size, size).to(device)  # 3x3のランダムなテンソル
    y = torch.randn(size, size).to(device)
    start_time = time.time()
    tensor = torch.mm(x, y)  # 行列の掛け算
    end_time = time.time()
    tensor = tensor.to(device)  # デバイスに転送
    print("Tensor created on device:", device)
    print(tensor)
    print("計算時間: {:.4f} 秒".format(end_time - start_time))
if __name__ == "__main__":
    check_cuda()