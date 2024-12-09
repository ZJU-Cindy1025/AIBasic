from Dataset import CifarTorchDataset, CifarSklearnDataset
import unittest
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkAgg')


class TestCifarDataset(unittest.TestCase):
    def test_CifarSklearnDataset1(self):
        """测试CifarSklearnDataset类
        训练集
        执行归一化
        执行数据旋转增广
        """
        file_path = 'fer2013_data/fer2013.csv'
        norm = True
        rotate = True
        train_val_test = 'train'
        dataset = CifarSklearnDataset(file_path, norm, rotate, train_val_test)
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(dataset.datas[0], cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 0].set_title(dataset.labels[0])
        ax[0, 1].imshow(dataset.datas[1], cmap='gray')
        ax[0, 1].axis('off')
        ax[0, 1].set_title(dataset.labels[1])
        ax[1, 0].imshow(dataset.datas[2], cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 0].set_title(dataset.labels[2])
        ax[1, 1].imshow(dataset.datas[3], cmap='gray')
        ax[1, 1].axis('off')
        ax[1, 1].set_title(dataset.labels[3])
        plt.show()
        plt.close()

    def test_CifarSklearnDataset2(self):
        """测试CifarSklearnDataset类
        验证集
        不执行归一化
        不执行数据旋转增广
        """
        file_path = 'fer2013_data/fer2013.csv'
        norm = False
        rotate = False
        train_val_test = 'val'
        dataset = CifarSklearnDataset(file_path, norm, rotate, train_val_test)
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(dataset.datas[0], cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 0].set_title(dataset.labels[0])
        ax[0, 1].imshow(dataset.datas[1], cmap='gray')
        ax[0, 1].axis('off')
        ax[0, 1].set_title(dataset.labels[1])
        ax[1, 0].imshow(dataset.datas[2], cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 0].set_title(dataset.labels[2])
        ax[1, 1].imshow(dataset.datas[3], cmap='gray')
        ax[1, 1].axis('off')
        ax[1, 1].set_title(dataset.labels[3])
        plt.show()
        plt.close()

    def test_CifarSklearnDataset3(self):
        """测试CifarSklearnDataset类
        测试集
        执行归一化
        执行数据旋转增广
        """
        file_path = 'fer2013_data/fer2013.csv'
        norm = True
        rotate = True
        train_val_test = 'test'
        dataset = CifarSklearnDataset(file_path, norm, rotate, train_val_test)
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(dataset.datas[0], cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 0].set_title(dataset.labels[0])
        ax[0, 1].imshow(dataset.datas[1], cmap='gray')
        ax[0, 1].axis('off')
        ax[0, 1].set_title(dataset.labels[1])
        ax[1, 0].imshow(dataset.datas[2], cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 0].set_title(dataset.labels[2])
        ax[1, 1].imshow(dataset.datas[3], cmap='gray')
        ax[1, 1].axis('off')
        ax[1, 1].set_title(dataset.labels[3])
        plt.show()
        plt.close()

    def test_CifarTorchDataset1(self):
        """测试CifarTorchDataset类
        训练集
        执行归一化
        执行数据旋转增广
        """
        file_path = 'fer2013_data/fer2013.csv'
        norm = True
        rotate = True
        train_val_test = 'train'
        dataset = CifarTorchDataset(file_path, norm, rotate, train_val_test)
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(dataset.datas[0], cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 0].set_title(dataset.labels[0])
        ax[0, 1].imshow(dataset.datas[1], cmap='gray')
        ax[0, 1].axis('off')
        ax[0, 1].set_title(dataset.labels[1])
        ax[1, 0].imshow(dataset.datas[2], cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 0].set_title(dataset.labels[2])
        ax[1, 1].imshow(dataset.datas[3], cmap='gray')
        ax[1, 1].axis('off')
        ax[1, 1].set_title(dataset.labels[3])
        plt.show()
        plt.close()

    def test_CifarTorchDataset2(self):
        """测试CifarTorchDataset类
        验证集
        不执行归一化
        不执行数据旋转增广
        """
        file_path = 'fer2013_data/fer2013.csv'
        norm = False
        rotate = False
        train_val_test = 'val'
        dataset = CifarTorchDataset(file_path, norm, rotate, train_val_test)
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(dataset.datas[0], cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 0].set_title(dataset.labels[0])
        ax[0, 1].imshow(dataset.datas[1], cmap='gray')
        ax[0, 1].axis('off')
        ax[0, 1].set_title(dataset.labels[1])
        ax[1, 0].imshow(dataset.datas[2], cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 0].set_title(dataset.labels[2])
        ax[1, 1].imshow(dataset.datas[3], cmap='gray')
        ax[1, 1].axis('off')
        ax[1, 1].set_title(dataset.labels[3])
        plt.show()
        plt.close()

    def test_CifarTorchDataset3(self):
        """测试CifarTorchDataset类
        测试集
        执行归一化
        执行数据旋转增广
        """
        file_path = 'fer2013_data/fer2013.csv'
        norm = True
        rotate = True
        train_val_test = 'test'
        dataset = CifarTorchDataset(file_path, norm, rotate, train_val_test)
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(dataset.datas[0], cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 0].set_title(dataset.labels[0])
        ax[0, 1].imshow(dataset.datas[1], cmap='gray')
        ax[0, 1].axis('off')
        ax[0, 1].set_title(dataset.labels[1])
        ax[1, 0].imshow(dataset.datas[2], cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 0].set_title(dataset.labels[2])
        ax[1, 1].imshow(dataset.datas[3], cmap='gray')
        ax[1, 1].axis('off')
        ax[1, 1].set_title(dataset.labels[3])
        plt.show()
        plt.close()


if __name__ == '__main__':
    unittest.main()
