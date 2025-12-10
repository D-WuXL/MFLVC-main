import torch.nn as nn
from torch.nn.functional import normalize
import torch

# 1. 保留原本的 MLP Encoder (為了兼容非圖像數據或其他視圖)
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

# 2. 插入您提供的 CNNEncoder (用於圖像數據)
class CNNEncoder(nn.Module):
    def __init__(self, input_channel, feature_dim):
        """
        input_channel: 圖片通道數 (Fashion-MNIST 為 1)
        feature_dim: 輸出特徵維度 (如 512)
        """
        super(CNNEncoder, self).__init__()

        self.features = nn.Sequential(
            # 輸入形狀: [Batch, 1, 28, 28]
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1), # -> [32, 28, 28]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> [32, 14, 14]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # -> [64, 14, 14]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> [64, 7, 7]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # -> [128, 7, 7]
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 全連接層將卷積特徵映射到目標維度
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 500),
            nn.ReLU(),
            nn.Linear(500, feature_dim)
        )

    def forward(self, x):
        # 1. 處理輸入形狀
        # 如果輸入是 [Batch, 784] 的向量，強制轉回圖片 [Batch, 1, 28, 28]
        if x.dim() == 2:
            batch_size = x.size(0)
            # 假設圖片是正方形，邊長為 sqrt(dim) -> sqrt(784)=28
            size = int(x.size(1) ** 0.5)
            x = x.view(batch_size, 1, size, size)

        # 2. 卷積提取特徵
        x = self.features(x)

        # 3. 拉平並映射
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

# 3. Decoder 保持不變 (原本代碼)
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

# 4. 修改 Network 類的初始化邏輯
class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            # --- 修改部分開始 ---
            # 判斷輸入維度，如果是 784 (Fashion-MNIST)，就自動切換成 CNNEncoder
            if input_size[v] == 784:
                # 這裡假設是灰階圖，input_channel=1
                self.encoders.append(CNNEncoder(1, feature_dim).to(device))
            else:
                # 其他維度繼續使用原來的 MLP Encoder
                self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            # --- 修改部分結束 ---

            # Decoder 保持使用 MLP
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
            # Varying the number of layers of W can obtain the representations with different shapes.
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.view = view

    def forward(self, xs):
        # 這裡不需要改，因為 CNNEncoder 的 forward 會自動處理維度
        hs = []
        qs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x) # 這裡會自動調用 CNNEncoder
            h = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.label_contrastive_module(z)
            xr = self.decoders[v](z) # Decoder 接收特徵 Z 還原回向量
            hs.append(h)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)
        return hs, qs, xrs, zs

    def forward_plot(self, xs):
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            h = self.feature_contrastive_module(z)
            hs.append(h)
        return zs, hs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.label_contrastive_module(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds