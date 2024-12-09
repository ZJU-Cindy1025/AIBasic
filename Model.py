from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.neural_network import MLPClassifier
import queue
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_curve, auc
import numpy as np
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class MachineLearningModel:
    """
    机器学习模型类（MLP）
    """

    def __init__(self, num_classes, hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, learning_rate_init, max_iter, early_stopping, train_dataset, val_dataset, test_dataset):
        """初始化函数
        :param num_classes: 输出类别数
        :param hidden_layer_sizes: 隐藏层大小
        :param activation: 激活函数
        :param solver: 优化器
        :param alpha: 正则化参数
        :param batch_size: 批大小
        :param learning_rate: 学习率
        :param learning_rate_init: 初始学习率
        :param max_iter: 最大迭代次数
        :param early_stopping: 是否使用EarlyStop
        :param train_dataset: 训练集
        :param val_dataset: 验证集
        :param test_dataset: 测试集
        """
        self.num_classes = num_classes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        # 构建模型
        self.mlp = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, solver=self.solver, alpha=self.alpha, batch_size=self.batch_size,
                                 learning_rate=self.learning_rate, learning_rate_init=self.learning_rate_init, max_iter=self.max_iter, early_stopping=self.early_stopping)
        self.log = queue.Queue()

    def Train(self):
        """
        训练函数
        """
        self.log.put('开始训练...')
        X = [d.flatten() for d in self.train_dataset.datas]
        y = self.train_dataset.labels
        self.mlp.fit(X, y)
        self.log.put('权重：'+str(self.mlp.coefs_))
        self.log.put('偏置：'+str(self.mlp.intercepts_))
        self.log.put('损失：'+str(self.mlp.loss_))
        self.log.put('迭代次数：'+str(self.mlp.n_iter_))
        self.Validate()

    def Validate(self) -> float:
        """
        验证函数
        """
        X = [d.flatten() for d in self.val_dataset.datas]
        y = self.val_dataset.labels
        score = self.mlp.score(X, y)
        self.log.put('验证集准确率：'+str(score))
        return score

    def Test(self) -> dict:
        """
        测试函数
        """
        self.log.put('开始测试...')
        X = [d.flatten() for d in self.test_dataset.datas]
        y = self.test_dataset.labels
        # 对每个类计算精确率和召回率
        Accuracy = [0 for i in range(self.num_classes)]
        Precision = [0 for i in range(self.num_classes)]
        Recall = [0 for i in range(self.num_classes)]
        F1 = [0 for i in range(self.num_classes)]
        TP = [0 for i in range(self.num_classes)]
        FP = [0 for i in range(self.num_classes)]
        TN = [0 for i in range(self.num_classes)]
        FN = [0 for i in range(self.num_classes)]
        for i in range(self.num_classes):
            for j in range(len(y)):
                if y[j] == i:
                    if self.mlp.predict([X[j]])[0] == i:
                        TP[i] += 1
                    else:
                        FN[i] += 1
                else:
                    if self.mlp.predict([X[j]])[0] == i:
                        FP[i] += 1
                    else:
                        TN[i] += 1
        for i in range(self.num_classes):
            if (TP[i]+FP[i]+FN[i] != 0):
                Accuracy[i] = TP[i]/(TP[i]+FP[i]+FN[i])
            if (TP[i]+FP[i] != 0):
                Precision[i] = TP[i]/(TP[i]+FP[i])
            if (TP[i]+FN[i] != 0):
                Recall[i] = TP[i]/(TP[i]+FN[i])
            if (Precision[i]+Recall[i] != 0):
                F1[i] = 2*Precision[i]*Recall[i]/(Precision[i]+Recall[i])
            self.log.put('类别'+str(i)+'：')
            self.log.put('准确率：'+str(Accuracy[i]))
            self.log.put('精确率：'+str(Precision[i]))
            self.log.put('召回率：'+str(Recall[i]))
            self.log.put('F1：'+str(F1[i]))
        # 绘制ROC曲线
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(
                (np.array(y) == i).astype(int), self.mlp.predict_proba(X)[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        for i in range(self.num_classes):
            plt.plot(fpr[i], tpr[i],
                     label='Class {0} ({1:0.2f})'
                     ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率')
        plt.ylabel('真正率')
        plt.title('ROC曲线')
        plt.legend()
        plt.show()
        plt.close()
        return {
            'Accuracy': Accuracy,
            'Precision': Precision,
            'Recall': Recall,
            'F1': F1
        }


class DeepLearningModel:
    """
    深度学习模型类
    """

    def __init__(self, num_classes, net, batch_size: int, epoch: int, shuffle: bool, Validate_frequency: int, lr: float, lr_decent: bool, device: torch.device, criterion, optimizer: str, OutputDir: str, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset):
        """初始化函数
        :param num_classes: 输出类别数
        :param net: 网络模型
        :param batch_size: 批大小
        :param epoch: 训练轮数
        :param shuffle: 是否打乱数据
        :param Validate_frequency: 验证频率，每多少个epoch验证一次，设置为0则不验证
        :param lr: 学习率
        :param lr_decent: 是否使用学习率下降
        :param device: 训练设备
        :param criterion: 损失函数
        :param optimizer: 优化器
        :param OutputDir: 模型输出路径
        :param train_dataset: 训练集
        :param val_dataset: 验证集
        :param test_dataset: 测试集
        """
        torch.cuda.empty_cache()
        self.num_classes = num_classes
        self.net = net
        self.batch_size = batch_size
        self.epoch = epoch
        self.shuffle = shuffle
        self.Validate_frequency = Validate_frequency
        self.lr = lr
        self.lr_decent = lr_decent
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.OutputDir = OutputDir
        self.Pause = False  # 是否暂停
        self.Restart = False  # 是否重新训练
        # 加载到设备
        self.net = self.net.to(self.device)
        self.criterion = self.criterion.to(self.device)
        # 准备数据集
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        # 归一化必须要一样
        assert self.train_dataset.norm == self.val_dataset.norm == self.test_dataset.norm
        # 加载数据集
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        # 用于输入到GUI窗口的日志
        self.log = queue.Queue()

    def Train(self) -> tuple:
        """
        训练函数
        返回值：嵌套元组
        ((训练损失表, 训练集精度表),验证集精度表)
        """
        self.log.put('开始训练...')
        torch.save(self.net.state_dict(), self.OutputDir+'/model_0.pth')
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        # 训练
        for epoch in range(self.epoch):
            # 暂停
            while (self.Pause):
                if (self.Restart):
                    self.net.load_state_dict(
                        torch.load(self.OutputDir+'/model_0.pth'))
                    return (0, 0), 0
                else:
                    time.sleep(0.1)
            match(self.optimizer):
                case 'Adam':
                    self.optimizer = torch.optim.Adam(
                        self.net.parameters(), lr=self.lr)
                case 'SGD':
                    self.optimizer = torch.optim.SGD(
                        self.net.parameters(), lr=self.lr)
                case 'RMSprop':
                    self.optimizer = torch.optim.RMSprop(
                        self.net.parameters(), lr=self.lr)
                case 'Adagrad':
                    self.optimizer = torch.optim.Adagrad(
                        self.net.parameters(), lr=self.lr)
                case 'Adadelta':
                    self.optimizer = torch.optim.Adadelta(
                        self.net.parameters(), lr=self.lr)
                case 'AdamW':
                    self.optimizer = torch.optim.AdamW(
                        self.net.parameters(), lr=self.lr)
                case 'SparseAdam':
                    self.optimizer = torch.optim.SparseAdam(
                        self.net.parameters(), lr=self.lr)
                case 'Adamax':
                    self.optimizer = torch.optim.Adamax(
                        self.net.parameters(), lr=self.lr)
                case 'ASGD':
                    self.optimizer = torch.optim.ASGD(
                        self.net.parameters(), lr=self.lr)
                case 'Rprop':
                    self.optimizer = torch.optim.Rprop(
                        self.net.parameters(), lr=self.lr)
                case 'LBFGS':
                    self.optimizer = torch.optim.LBFGS(
                        self.net.parameters(), lr=self.lr)
                case 'RMSprop':
                    self.optimizer = torch.optim.RMSprop(
                        self.net.parameters(), lr=self.lr)
                case 'Rprop':
                    self.optimizer = torch.optim.Rprop(
                        self.net.parameters(), lr=self.lr)
                case 'LBFGS':
                    self.optimizer = torch.optim.LBFGS(
                        self.net.parameters(), lr=self.lr)
                case 'SparceAdam':
                    self.optimizer = torch.optim.SparseAdam(
                        self.net.parameters(), lr=self.lr)
                case 'NAdam':
                    self.optimizer = torch.optim.NAdam(
                        self.net.parameters(), lr=self.lr)
                case 'FTRL':
                    self.optimizer = torch.optim.FTRL(
                        self.net.parameters(), lr=self.lr)

            # 训练模型
            self.net.train()
            train_loss = 0
            train_accuracy = 0
            for data, label in self.train_dataloader:
                data, label = data.to(self.device), label.to(self.device)
                output = self.net(data)
                loss = self.criterion(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_accuracy += (output.argmax(1) == label).sum().item()
            train_loss /= len(self.train_dataloader)
            train_accuracy /= len(self.train_dataset)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            # 记录训练日志
            self.log.put(
                f'Epoch {epoch+1}/{self.epoch}\n训练集损失: {train_loss}\n训练集精度: {train_accuracy}')
            if (self.Validate_frequency != 0 and (epoch+1) % self.Validate_frequency == 0):
                val_accuracies.append(self.Validate())
            if self.lr_decent:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.epoch//10, gamma=0.1)
                scheduler.step()
            torch.save(self.net.state_dict(),
                       self.OutputDir+'/model_%d.pth' % (epoch+1))
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].plot(train_losses)
        ax[0].set_title('训练损失')
        ax[1].plot(train_accuracies)
        ax[1].set_title('训练集精度')
        ax[2].plot(val_accuracies)
        ax[2].set_title('验证集精度')
        plt.tight_layout()
        plt.show()
        plt.close()
        return (train_losses, train_accuracies), val_accuracies

    def Validate(self) -> float:
        """
        验证函数
        返回值：验证准确率
        """
        self.log.put('开始验证...')
        self.net.eval()
        val_acc = 0
        with torch.no_grad():
            for data, label in self.val_dataloader:
                data, label = data.to(self.device), label.to(self.device)
                output = self.net(data)
                val_acc += (output.argmax(1) == label).sum().item()
        val_acc /= len(self.val_dataset)
        self.log.put(f'验证集精度: {val_acc}')
        return val_acc

    def Test(self, modelnum: int) -> dict:
        """
        测试函数
        返回值：测试准确率
        """
        self.log.put('开始测试...')
        self.net.eval()
        model = torch.load(self.OutputDir+'/model_%d.pth' % modelnum)
        test_acc = 0
        with torch.no_grad():
            self.net.load_state_dict(model)
            for data, label in self.test_dataloader:
                if (self.Restart == True):
                    return {}
                data, label = data.to(self.device), label.to(self.device)
                output = self.net(data)
                test_acc += (output.argmax(1) == label).sum().item()
        test_acc /= len(self.test_dataset)
        # 计算每个类别的精确率和召回率
        Accuracy = [0 for i in range(self.num_classes)]
        Precision = [0 for i in range(self.num_classes)]
        Recall = [0 for i in range(self.num_classes)]
        F1 = [0 for i in range(self.num_classes)]
        TP = [0 for i in range(self.num_classes)]
        FP = [0 for i in range(self.num_classes)]
        FN = [0 for i in range(self.num_classes)]
        TN = [0 for i in range(self.num_classes)]
        for i in range(self.num_classes):
            if (self.Restart == True):
                return {}
            for j in range(len(self.test_dataset)):
                data, label = self.test_dataset[j]
                data, label = data.unsqueeze(0).to(
                    self.device), label.to(self.device)
                output = self.net(data)
                if label == i:
                    if output.argmax(1) == label:
                        TP[i] += 1
                    else:
                        FN[i] += 1
                else:
                    if output.argmax(1) == label:
                        FP[i] += 1
                    else:
                        TN[i] += 1
        for i in range(self.num_classes):
            if (TP[i]+FP[i]+FN[i] != 0):
                Accuracy[i] = TP[i]/(TP[i]+FP[i]+FN[i])
            if (TP[i]+FP[i] != 0):
                Precision[i] = TP[i]/(TP[i]+FP[i])
            if (TP[i]+FN[i] != 0):
                Recall[i] = TP[i]/(TP[i]+FN[i])
            if (Precision[i]+Recall[i] != 0):
                F1[i] = 2*Precision[i]*Recall[i]/(Precision[i]+Recall[i])
            self.log.put('类别'+str(i)+'：')
            self.log.put('准确率：'+str(Accuracy[i]))
            self.log.put('精确率：'+str(Precision[i]))
            self.log.put('召回率：'+str(Recall[i]))
            self.log.put('F1：'+str(F1[i]))
        # 绘制ROC曲线
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        all_labels = []
        all_outputs = []
        for data, label in self.test_dataloader:
            if (self.Restart == True):
                return {}
            data, label = data.to(self.device), label.to(self.device)
            output = self.net(data)
            all_labels.extend(label.cpu().numpy())
            all_outputs.extend(output.cpu().detach().numpy())
        all_labels = np.array(all_labels)
        all_outputs = np.array(all_outputs)
        for i in range(self.num_classes):
            if (self.Restart == True):
                return {}
            fpr[i], tpr[i], _ = roc_curve(
                (all_labels == i).astype(int), all_outputs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        for i in range(self.num_classes):
            if (self.Restart == True):
                return {}
            plt.plot(fpr[i], tpr[i],
                     label='Class {0} ({1:0.2f})'
                     ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率')
        plt.ylabel('真正率')
        plt.title('ROC曲线')
        plt.legend()
        plt.show()
        plt.close()
        return {
            'Accuracy': test_acc,
            'Precision': Precision,
            'Recall': Recall,
            'F1': F1
        }
