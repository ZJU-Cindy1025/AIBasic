import unittest
import torch
import torch.nn as nn
from Model import DeepLearningModel, MachineLearningModel
from Net import AlexNet, MLP
from Dataset import CifarSklearnDataset, CifarTorchDataset


class TestModel(unittest.TestCase):
    def test_SklearnMLP(self):
        """测试基于机器学习的MLP网络的模型训练
        隐藏层大小：(128,128)
        激活函数：ReLU
        优化器：Adam
        学习率：constant
        批大小：auto
        训练轮数：100
        是否打乱数据：True
        是否使用Earlystop：False
        训练集：'fer2013_data/fer2013.csv'，执行归一化，不执行数据旋转增广
        验证集：'fer2013_data/fer2013.csv'，执行归一化，不执行数据旋转增广
        测试集：'fer2013_data/fer2013.csv'，执行归一化，不执行数据旋转增广
        """
        # 数据集
        train_dataset = CifarSklearnDataset(
            file_path='fer2013_data/fer2013.csv', norm=True, rotate=False, train_val_test='train')
        val_dataset = CifarSklearnDataset(
            file_path='fer2013_data/fer2013.csv', norm=True, rotate=False, train_val_test='val')
        test_dataset = CifarSklearnDataset(
            file_path='fer2013_data/fer2013.csv', norm=True, rotate=False, train_val_test='test')
        # 模型
        model = MachineLearningModel(hidden_layer_sizes=(128, 128), num_classes=7, activation='relu', solver='adam', learning_rate='constant', batch_size='auto', max_iter=100,
                                     early_stopping=False, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, alpha=0.0001, learning_rate_init=0.001)
        # 训练
        model.Train()
        print(model.Validate())
        print(model.Test())

    def test_TorchMLP(self):
        """测试MLP网络的模型训练
        网络模型：MLP，输出类别数：7，MLP总层数：3，每层MLP的神经元数量：[512, 128, 512, 128]
        激活函数：ReLU
        批大小：64
        训练轮数：10
        是否打乱数据：True
        验证频率：2
        学习率：0.001
        训练设备：CUDA
        损失函数：CrossEntropyLoss
        优化器：Adam
        模型输出路径：'model/'       
        训练集：'fer2013_data/fer2013.csv'，执行归一化，不执行数据旋转增广
        验证集：'fer2013_data/fer2013.csv'，执行归一化，不执行数据旋转增广
        测试集：'fer2013_data/fer2013.csv' ，执行归一化，不执行数据旋转增广
        """
        # 网络模型：MLP
        net = MLP(num_classes=7, linear_num=[
                  512, 128, 512, 128], activation=nn.ReLU())
        # 模型
        train_dataset = CifarTorchDataset(
            file_path='fer2013_data/fer2013.csv', norm=True, rotate=False, train_val_test='train')
        val_dataset = CifarTorchDataset(
            file_path='fer2013_data/fer2013.csv', norm=True, rotate=False, train_val_test='val')
        test_dataset = CifarTorchDataset(
            file_path='fer2013_data/fer2013.csv', norm=True, rotate=False, train_val_test='test')
        # 模型
        model = DeepLearningModel(net=net, num_classes=7, batch_size=64, epoch=10, shuffle=True, Validate_frequency=2, lr=0.001, lr_decent=True, device=torch.device('cuda'), criterion=nn.CrossEntropyLoss(
        ), optimizer='Adam', OutputDir='model', train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)
        # 训练
        model.Train()
        print(model.Validate())
        print(model.Test(modelnum=10))

    def test_TorchAlexNet(self):
        """测试AlexNet网络的模型训练
        网络模型：AlexNet，输出类别数：7，全连接层神经元数：128，Dropout概率：0.5，激活函数：ReLU
        批大小：64
        训练轮数：10
        是否打乱数据：True
        验证频率：1
        学习率：0.001
        训练设备：CUDA
        损失函数：CrossEntropyLoss
        优化器：Adam
        模型输出路径：'model/'
        训练集：'fer2013_data/fer2013.csv'，执行归一化，不执行数据旋转增广
        验证集：'fer2013_data/fer2013.csv'，执行归一化，不执行数据旋转增广
        测试集：'fer2013_data/fer2013.csv'，执行归一化，不执行数据旋转增广
        """
        # 网络模型：AlexNet
        net = AlexNet(num_classes=7, linear_num=128,
                      p=0.5, activation=nn.ReLU())
        # 模型
        train_dataset = CifarTorchDataset(
            file_path='fer2013_data/fer2013.csv', norm=True, rotate=False, train_val_test='train')
        val_dataset = CifarTorchDataset(
            file_path='fer2013_data/fer2013.csv', norm=True, rotate=False, train_val_test='val')
        test_dataset = CifarTorchDataset(
            file_path='fer2013_data/fer2013.csv', norm=True, rotate=False, train_val_test='test')
        # 模型
        model = DeepLearningModel(net=net, num_classes=7, batch_size=64, epoch=10, shuffle=True, Validate_frequency=1, lr=0.001, lr_decent=True, device=torch.device('cuda'), criterion=nn.CrossEntropyLoss(
        ), optimizer='Adam', OutputDir='model', train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)
        # 训练
        model.Train()
        print(model.Validate())
        print(model.Test(modelnum=10))


if (__name__ == '__main__'):
    unittest.main()
