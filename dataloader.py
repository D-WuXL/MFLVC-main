from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    # --- 新增這段配置 ---
    elif dataset == "Fashion-TTAE":
        dataset = FashionTTAE('./data/', sample_size=2000)
        dims = [784, 784]  # 兩個視圖都是 784 維向量
        view = 2  # 只有 2 個視圖
        data_size = 2000  # 樣本數
        class_num = 10
    # ------------------
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num


#————————————————————————————————————————————
#按照TTAE论文要求进行改变
#————————————————————————————————————————————
import scipy.io
import scipy.ndimage
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class FashionTTAE(Dataset):
    def __init__(self, path, sample_size=2000):
        """
        Fashion-MNIST TTAE版数据集 (适配 MLP 全连接网络)
        View 1: 原始数据 (拉平为 784 维)
        View 2: 边缘特征 (拉平为 784 维)
        """
        # 1. 加载原始数据
        data = scipy.io.loadmat(path + 'Fashion.mat')

        full_x = data['X1'].astype(np.float32)
        full_y = data['Y'].astype(np.int32).reshape(-1)

        # 2. 随机采样 2000 个样本
        np.random.seed(10)
        total_samples = full_x.shape[0]
        if sample_size > total_samples:
            sample_size = total_samples
        indices = np.random.choice(total_samples, sample_size, replace=False)

        x_sampled = full_x[indices]
        self.y = full_y[indices]

        # 3. 【核心修正】强制拉平为 2维 [N, 784]
        # 无论原始数据是图片还是向量，统统拉平成 (N, 784)
        if x_sampled.ndim > 2:
            x_sampled = x_sampled.reshape(x_sampled.shape[0], -1)

        # 确保是 784 维
        if x_sampled.shape[1] != 784:
            # 如果不是 784，尝试强制 reshape (前提是总像素数对得上)
            try:
                x_sampled = x_sampled.reshape(x_sampled.shape[0], 784)
            except ValueError:
                print(f"警告: 数据维度异常 {x_sampled.shape}, 无法 Reshape 为 784")

        # 4. 构造视图
        self.v1 = x_sampled
        # 生成边缘特征 (函数内部会处理图片转换，最后返回拉平的向量)
        self.v2 = self._generate_edge_view(x_sampled)

        # 5. 归一化 (MLP 版本推荐使用 MinMaxScaler)
        scaler = MinMaxScaler()
        self.v1 = scaler.fit_transform(self.v1)
        self.v2 = scaler.fit_transform(self.v2)

    def _generate_edge_view(self, x_vecs):
        """
        输入: x_vecs [N, 784]
        输出: edge_vecs [N, 784]
        """
        edge_data = []
        N = x_vecs.shape[0]
        for i in range(N):
            # 1. 还原为图片以提取边缘
            img = x_vecs[i].reshape(28, 28)

            # 2. Sobel 边缘提取
            sx = scipy.ndimage.sobel(img, axis=0, mode='constant')
            sy = scipy.ndimage.sobel(img, axis=1, mode='constant')
            edge_img = np.hypot(sx, sy)

            # 3. 【关键】提取完边缘后，再次拉平回向量
            edge_data.append(edge_img.reshape(-1))

        return np.array(edge_data, dtype=np.float32)

    def __len__(self):
        return self.v1.shape[0]

    def __getitem__(self, idx):
        # 返回: [View1, View2]
        # 此时形状都是 [784]，符合 Linear 层的输入要求
        return [torch.from_numpy(self.v1[idx]), torch.from_numpy(self.v2[idx])], \
            self.y[idx], torch.from_numpy(np.array(idx)).long()