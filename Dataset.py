import torch
from torch.utils.data import Dataset
import csv
import numpy as np
import pandas as pd


class CifarSklearnDataset:
    """
    基于Scikit-learn的Cifar数据集
    """

    def __init__(self, file_path: str, norm: bool, rotate: bool, train_val_test: str):
        """初始化函数
        :param file_path: 数据集文件路径
        :param norm: 是否进行归一化
        :param rotate: 是否进行数据旋转增广
        :param train_val_test: 训练集、验证集、测试集
        """
        # 参数设置
        self.norm = norm
        self.rotate = rotate
        df = pd.read_csv(file_path)
        match(train_val_test):
            case 'train':
                df = df[df['Usage'] == 'Training']
            case 'val':
                df = df[df['Usage'] == 'PrivateTest']
            case 'test':
                df = df[df['Usage'] == 'PublicTest']
        self.datas = df['pixels'].values
        if (self.norm):
            self.datas = [np.array(list(map(int, i.split()))).reshape(
                (48, 48))/255.0 for i in self.datas]
        else:
            self.datas = [np.array(list(map(int, i.split()))).reshape(
                (48, 48)) for i in self.datas]
        self.labels = df['emotion'].values
        if (self.rotate):
            rotate_90 = [np.rot90(i, 1) for i in self.datas]
            rotate_180 = [np.rot90(i, 2) for i in self.datas]
            rotate_270 = [np.rot90(i, 3) for i in self.datas]
            self.datas = self.datas+rotate_90+rotate_180+rotate_270
            self.labels = [i for i in self.labels]*4


class CifarTorchDataset(Dataset):
    """
    基于PyTorch的CifarDataset数据集
    """

    def __init__(self, file_path: str, norm: bool, rotate: bool, train_val_test: str):
        """初始化函数
        :param file_path: 数据集文件路径
        :param norm: 是否进行归一化
        :param rotate: 是否进行数据旋转增广
        :param train_val_test: 训练集、验证集、测试集
        """
        # 参数设置
        self.norm = norm
        self.rotate = rotate
        self.datas = []
        self.labels = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                # line[0]：标签 line[1]：数据（48*48） line[2]：Training，PublicTest，PrivateTest
                match(train_val_test):
                    case 'train':
                        if (line[-1] == 'Training'):
                            # 训练集
                            data = np.array(
                                list(map(int, line[1].split()))).reshape((48, 48))
                            if (self.norm):
                                # 归一化
                                data = data/255.0
                            if (self.rotate):
                                # 数据增广
                                self.datas.append(data)
                                self.datas.append(np.rot90(data, 1))
                                self.datas.append(np.rot90(data, 2))
                                self.datas.append(np.rot90(data, 3))
                                self.labels.append(int(line[0]))
                                self.labels.append(int(line[0]))
                                self.labels.append(int(line[0]))
                                self.labels.append(int(line[0]))
                            else:
                                self.datas.append(data)
                                self.labels.append(int(line[0]))
                    case 'val':
                        if (line[-1] == 'PrivateTest'):
                            # 验证集
                            data = np.array(
                                list(map(int, line[1].split()))).reshape((48, 48))
                            if (self.norm):
                                # 归一化
                                data = data/255.0
                            if (self.rotate):
                                # 数据增广
                                self.datas.append(data)
                                self.datas.append(np.rot90(data, 1))
                                self.datas.append(np.rot90(data, 2))
                                self.datas.append(np.rot90(data, 3))
                                self.labels.append(int(line[0]))
                                self.labels.append(int(line[0]))
                                self.labels.append(int(line[0]))
                                self.labels.append(int(line[0]))
                            else:
                                self.datas.append(data)
                                self.labels.append(int(line[0]))
                    case 'test':
                        if (line[-1] == 'PublicTest'):
                            # 测试集
                            data = np.array(
                                list(map(int, line[1].split()))).reshape((48, 48))
                            if (self.norm):
                                # 归一化
                                data = data/255.0
                            if (self.rotate):
                                # 数据增广
                                self.datas.append(data)
                                self.datas.append(np.rot90(data, 1))
                                self.datas.append(np.rot90(data, 2))
                                self.datas.append(np.rot90(data, 3))
                                self.labels.append(int(line[0]))
                                self.labels.append(int(line[0]))
                                self.labels.append(int(line[0]))
                                self.labels.append(int(line[0]))
                            else:
                                self.datas.append(data)
                                self.labels.append(int(line[0]))

    def __len__(self):
        """返回数据集大小"""
        return len(self.datas)

    def __getitem__(self, idx):
        """根据idx返回一条数据"""
        return torch.tensor(self.datas[idx].copy(), dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.long)
