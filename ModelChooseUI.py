import tkinter as tk
import tkinter.ttk as ttk
from Model import MachineLearningModel, DeepLearningModel
from tkinter import filedialog, messagebox
from Net import MLP, AlexNet
import torch.nn as nn
from Dataset import CifarSklearnDataset, CifarTorchDataset
from ModelTrainUI import ModelTrainUI


class ModelChooseUI(tk.Frame):
    """
    模型选择界面
    """

    def __init__(self, master):
        super().__init__(master)
        self.pack()
        self.create_widgets()
        self.ChooseModel()

    def create_widgets(self):
        self.ModelChoose = tk.IntVar()
        self.ModelChoose.set(1)
        # MLP(Scikit-learn)部分
        self.RadioButton1 = tk.Radiobutton(
            self, text='MLP(Scikit-learn)', value=1, variable=self.ModelChoose, command=self.ChooseModel)
        self.RadioButton1.grid(row=0, column=1)
        # 隐藏层大小（输入框）
        self.value11 = tk.StringVar()
        self.value11.set('128 128')
        self.Label11 = tk.Label(self, text='隐藏层大小')
        self.Label11.grid(row=1, column=0, padx=5, pady=5)
        self.Entry11 = tk.Entry(self, textvariable=self.value11)
        self.Entry11.grid(row=1, column=1, padx=5, pady=5)
        # 激活函数（下拉框）
        self.value12 = tk.StringVar()
        self.value12.set('relu')
        self.Label12 = tk.Label(self, text='激活函数')
        self.Label12.grid(row=2, column=0, padx=5, pady=5)
        self.ComboBox12 = ttk.Combobox(self, textvariable=self.value12, values=(
            'identity', 'logistic', 'tanh', 'relu'), state="readonly")
        self.ComboBox12.grid(row=2, column=1, padx=5, pady=5)
        # 优化器（下拉框）
        self.value13 = tk.StringVar()
        self.value13.set('adam')
        self.Label13 = tk.Label(self, text='优化器')
        self.Label13.grid(row=3, column=0, padx=5, pady=5)
        self.ComboBox13 = ttk.Combobox(self, textvariable=self.value13, values=(
            'adam', 'lbfgs', 'sgd'), state="readonly")
        self.ComboBox13.grid(row=3, column=1, padx=5, pady=5)
        # 正则化参数（输入框）
        self.value14 = tk.DoubleVar()
        self.value14.set(0.0001)
        self.Label14 = tk.Label(self, text='正则化参数')
        self.Label14.grid(row=4, column=0, padx=5, pady=5)
        self.Entry14 = tk.Entry(self, textvariable=self.value14)
        self.Entry14.grid(row=4, column=1, padx=5, pady=5)
        # 初始学习率（输入框）
        self.value15 = tk.DoubleVar()
        self.value15.set(0.001)
        self.Label15 = tk.Label(self, text='初始学习率')
        self.Label15.grid(row=5, column=0, padx=5, pady=5)
        self.Entry15 = tk.Entry(self, textvariable=self.value15)
        self.Entry15.grid(row=5, column=1, padx=5, pady=5)
        # 学习率变化方式（下拉框）
        self.value16 = tk.StringVar()
        self.value16.set('constant')
        self.Label16 = tk.Label(self, text='学习率变化方式')
        self.Label16.grid(row=6, column=0, padx=5, pady=5)
        self.ComboBox16 = ttk.Combobox(self, textvariable=self.value16, values=(
            'constant', 'invscaling', 'adaptive'), state="readonly")
        self.ComboBox16.grid(row=6, column=1, padx=5, pady=5)
        # 最大迭代次数（输入框）
        self.value17 = tk.IntVar()
        self.value17.set(200)
        self.Label17 = tk.Label(self, text='最大迭代次数')
        self.Label17.grid(row=7, column=0, padx=5, pady=5)
        self.Entry17 = tk.Entry(self, textvariable=self.value17)
        self.Entry17.grid(row=7, column=1, padx=5, pady=5)
        # 早停参数（下拉框）
        self.value18 = tk.BooleanVar()
        self.value18.set(False)
        self.Label18 = tk.Label(self, text='早停参数')
        self.Label18.grid(row=8, column=0, padx=5, pady=5)
        self.Entry18 = ttk.Combobox(
            self, textvariable=self.value18, values=(True, False), state="readonly")
        self.Entry18.grid(row=8, column=1, padx=5, pady=5)
        # 批大小（输入框）
        self.value19 = tk.StringVar()
        self.value19.set('auto')
        self.Label19 = tk.Label(self, text='批大小')
        self.Label19.grid(row=9, column=0, padx=5, pady=5)
        self.Entry19 = tk.Entry(self, textvariable=self.value19)
        self.Entry19.grid(row=9, column=1, padx=5, pady=5)

        # MLP(Pytorch部分)
        self.RadioButton2 = tk.Radiobutton(
            self, text='MLP(Pytorch)', value=2, variable=self.ModelChoose, command=self.ChooseModel)
        self.RadioButton2.grid(row=0, column=3, padx=5, pady=5)
        # 每层神经元数量（输入框）
        self.value21 = tk.StringVar()
        self.value21.set('512 128 512 128')
        self.Label21 = tk.Label(self, text='每层神经元数量')
        self.Label21.grid(row=1, column=2, padx=5, pady=5)
        self.Entry21 = tk.Entry(self, textvariable=self.value21)
        self.Entry21.grid(row=1, column=3, padx=5, pady=5)
        # 激活函数（下拉框）
        self.value22 = tk.StringVar()
        self.value22.set('ReLU')
        self.Label22 = tk.Label(self, text='激活函数')
        self.Label22.grid(row=2, column=2, padx=5, pady=5)
        self.ComboBox22 = ttk.Combobox(self, textvariable=self.value22, values=(
            'ReLU', 'Leaky ReLU', 'ELU', 'SELU', 'Sigmoid', 'Tanh', 'Softmax', 'Softplus', 'Hardtanh'), state="readonly")
        self.ComboBox22.grid(row=2, column=3, padx=5, pady=5)
        # 批大小（输入框）
        self.value23 = tk.IntVar()
        self.value23.set(64)
        self.Label23 = tk.Label(self, text='批大小')
        self.Label23.grid(row=3, column=2, padx=5, pady=5)
        self.Entry23 = tk.Entry(self, textvariable=self.value23)
        self.Entry23.grid(row=3, column=3, padx=5, pady=5)
        # 训练轮数（输入框）
        self.value24 = tk.IntVar()
        self.value24.set(100)
        self.Label24 = tk.Label(self, text='训练轮数')
        self.Label24.grid(row=4, column=2, padx=5, pady=5)
        self.Entry24 = tk.Entry(self, textvariable=self.value24)
        self.Entry24.grid(row=4, column=3, padx=5, pady=5)
        # 是否打乱数据（下拉框）
        self.value25 = tk.BooleanVar()
        self.value25.set(False)
        self.Label25 = tk.Label(self, text='是否打乱数据')
        self.Label25.grid(row=5, column=2, padx=5, pady=5)
        self.ComboBox25 = ttk.Combobox(
            self, textvariable=self.value25, values=(True, False), state="readonly")
        self.ComboBox25.grid(row=5, column=3, padx=5, pady=5)
        # 验证频率（输入框）
        self.value26 = tk.IntVar()
        self.value26.set(1)
        self.Label26 = tk.Label(self, text='验证频率')
        self.Label26.grid(row=6, column=2, padx=5, pady=5)
        self.Entry26 = tk.Entry(self, textvariable=self.value26)
        self.Entry26.grid(row=6, column=3, padx=5, pady=5)
        # 初始学习率（输入框）
        self.value27 = tk.DoubleVar()
        self.value27.set(0.001)
        self.Label27 = tk.Label(self, text='初始学习率')
        self.Label27.grid(row=7, column=2, padx=5, pady=5)
        self.Entry27 = tk.Entry(self, textvariable=self.value27)
        self.Entry27.grid(row=7, column=3, padx=5, pady=5)
        # 学习率变化（下拉框）
        self.value28 = tk.BooleanVar()
        self.value28.set(False)
        self.Label28 = tk.Label(self, text='学习率变化')
        self.Label28.grid(row=8, column=2, padx=5, pady=5)
        self.ComboBox28 = ttk.Combobox(
            self, textvariable=self.value28, values=(True, False), state="readonly")
        self.ComboBox28.grid(row=8, column=3, padx=5, pady=5)
        # 训练设备（下拉框）
        self.value29 = tk.StringVar()
        self.value29.set('cpu')
        self.Label29 = tk.Label(self, text='训练设备')
        self.Label29.grid(row=9, column=2, padx=5, pady=5)
        self.ComboBox29 = ttk.Combobox(
            self, textvariable=self.value29, values=('cpu', 'cuda'), state="readonly")
        self.ComboBox29.grid(row=9, column=3, padx=5, pady=5)
        # 损失函数（下拉框）
        self.value210 = tk.StringVar()
        self.value210.set('CrossEntropyLoss')
        self.Label210 = tk.Label(self, text='损失函数')
        self.Label210.grid(row=10, column=2, padx=5, pady=5)
        self.ComboBox210 = ttk.Combobox(self, textvariable=self.value210, values=('CrossEntropyLoss', 'MSELoss', 'BCELoss', 'BCEWithLogitsLoss',
                                        'TripletMarginLoss', 'KLDivLoss', 'CosineEmbeddingLoss', 'MarginRankingLoss', 'PoissonNLLLoss', 'SoftMarginLoss'), state="readonly")
        self.ComboBox210.grid(row=10, column=3, padx=5, pady=5)
        # 优化器（下拉框）
        self.value211 = tk.StringVar()
        self.value211.set('Adam')
        self.Label211 = tk.Label(self, text='优化器')
        self.Label211.grid(row=11, column=2, padx=5, pady=5)
        self.ComboBox211 = ttk.Combobox(self, textvariable=self.value211, values=(
            'Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'AdamW', 'AdamW', 'SparseAdam', 'Adamax', 'ASGD', 'Rprop', 'LBFGS', 'RMSprop', 'Rprop', 'LBFGS', 'SparceAdam', 'NAdam', 'FTRL'), state="readonly")
        self.ComboBox211.grid(row=11, column=3, padx=5, pady=5)

        # AlexNet(Pytorch部分)
        self.RadioButton3 = tk.Radiobutton(
            self, text='AlexNet(Pytorch)', value=3, variable=self.ModelChoose, command=self.ChooseModel)
        self.RadioButton3.grid(row=0, column=5, padx=5, pady=5)
        # 全连接层神经元数量（输入框）
        self.value31 = tk.IntVar()
        self.value31.set(128)
        self.Label31 = tk.Label(self, text='全连接层神经元数量')
        self.Label31.grid(row=1, column=4, padx=5, pady=5)
        self.Entry31 = tk.Entry(self, textvariable=self.value31)
        self.Entry31.grid(row=1, column=5, padx=5, pady=5)
        # 激活函数（下拉框）
        self.value32 = tk.StringVar()
        self.value32.set('ReLU')
        self.Label32 = tk.Label(self, text='激活函数')
        self.Label32.grid(row=2, column=4, padx=5, pady=5)
        self.ComboBox32 = ttk.Combobox(self, values=(
            'ReLU', 'Leaky ReLU', 'ELU', 'SELU', 'Sigmoid', 'Tanh', 'Softmax', 'Softplus', 'Hardtanh'), state="readonly", textvariable=self.value32)
        self.ComboBox32.grid(row=2, column=5, padx=5, pady=5)
        # Dropout概率（输入框）
        self.value33 = tk.DoubleVar()
        self.value33.set(0.5)
        self.Label33 = tk.Label(self, text='Dropout概率')
        self.Label33.grid(row=3, column=4, padx=5, pady=5)
        self.Entry33 = tk.Entry(self, textvariable=self.value33)
        self.Entry33.grid(row=3, column=5, padx=5, pady=5)
        # 批大小（输入框）
        self.value34 = tk.IntVar()
        self.value34.set(64)
        self.Label34 = tk.Label(self, text='批大小')
        self.Label34.grid(row=4, column=4, padx=5, pady=5)
        self.Entry34 = tk.Entry(self, textvariable=self.value34)
        self.Entry34.grid(row=4, column=5, padx=5, pady=5)
        # 训练轮数（输入框）
        self.value35 = tk.IntVar()
        self.value35.set(100)
        self.Label35 = tk.Label(self, text='训练轮数')
        self.Label35.grid(row=5, column=4, padx=5, pady=5)
        self.Entry35 = tk.Entry(self, textvariable=self.value35)
        self.Entry35.grid(row=5, column=5, padx=5, pady=5)
        # 是否打乱数据（下拉框）
        self.value36 = tk.BooleanVar()
        self.value36.set(False)
        self.Label36 = tk.Label(self, text='是否打乱数据')
        self.Label36.grid(row=6, column=4, padx=5, pady=5)
        self.ComboBox36 = ttk.Combobox(
            self, textvariable=self.value36, values=(True, False), state="readonly")
        self.ComboBox36.grid(row=6, column=5, padx=5, pady=5)
        # 验证频率（输入框）
        self.value37 = tk.IntVar()
        self.value37.set(1)
        self.Label37 = tk.Label(self, text='验证频率')
        self.Label37.grid(row=7, column=4, padx=5, pady=5)
        self.Entry37 = tk.Entry(self, textvariable=self.value37)
        self.Entry37.grid(row=7, column=5, padx=5, pady=5)
        # 初始学习率
        self.value38 = tk.DoubleVar()
        self.value38.set(0.001)
        self.Label38 = tk.Label(self, text='初始学习率')
        self.Label38.grid(row=8, column=4, padx=5, pady=5)
        self.Entry38 = tk.Entry(self, textvariable=self.value38)
        self.Entry38.grid(row=8, column=5, padx=5, pady=5)
        # 学习率变化（下拉框）
        self.value39 = tk.BooleanVar()
        self.value39.set(False)
        self.Label39 = tk.Label(self, text='学习率变化')
        self.Label39.grid(row=9, column=4, padx=5, pady=5)
        self.ComboBox39 = ttk.Combobox(
            self, textvariable=self.value39, values=(True, False), state="readonly")
        self.ComboBox39.grid(row=9, column=5, padx=5, pady=5)
        # 训练设备（下拉框）
        self.value310 = tk.StringVar()
        self.value310.set('cpu')
        self.Label310 = tk.Label(self, text='训练设备')
        self.Label310.grid(row=10, column=4, padx=5, pady=5)
        self.ComboBox310 = ttk.Combobox(
            self, textvariable=self.value310, values=('cpu', 'cuda'), state="readonly")
        self.ComboBox310.grid(row=10, column=5, padx=5, pady=5)
        # 损失函数（下拉框）
        self.value311 = tk.StringVar()
        self.value311.set('CrossEntropyLoss')
        self.Label311 = tk.Label(self, text='损失函数')
        self.Label311.grid(row=11, column=4, padx=5, pady=5)
        self.ComboBox311 = ttk.Combobox(self, textvariable=self.value311, values=('CrossEntropyLoss', 'MSELoss', 'BCELoss', 'BCEWithLogitsLoss',
                                        'TripletMarginLoss', 'KLDivLoss', 'CosineEmbeddingLoss', 'MarginRankingLoss', 'PoissonNLLLoss', 'SoftMarginLoss'), state="readonly")
        self.ComboBox311.grid(row=11, column=5, padx=5, pady=5)
        # 优化器（下拉框）
        self.value312 = tk.StringVar()
        self.value312.set('Adam')
        self.Label312 = tk.Label(self, text='优化器')
        self.Label312.grid(row=12, column=4, padx=5, pady=5)
        self.ComboBox312 = ttk.Combobox(self, textvariable=self.value312, values=(
            'Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'AdamW', 'AdamW', 'SparseAdam', 'Adamax', 'ASGD', 'Rprop', 'LBFGS', 'RMSprop', 'Rprop', 'LBFGS', 'SparceAdam', 'NAdam', 'FTRL'), state="readonly")
        self.ComboBox312.grid(row=12, column=5, padx=5, pady=5)

        # 选择数据集
        self.Label1 = tk.Label(self, text='数据集目录')
        self.Label1.grid(row=13, column=1, padx=5, pady=5)
        self.DatasetDir = tk.StringVar()
        # 预设
        # self.DatasetDir.set('fer2013_data/fer2013.csv')
        self.Entry1 = tk.Entry(self, textvariable=self.DatasetDir)
        self.Entry1.grid(row=13, column=2, padx=5, pady=5)
        self.Button1 = tk.Button(self, text='选择数据集', command=self.AskDataset)
        self.Button1.grid(row=13, column=3, padx=5, pady=5)
        # 数据集归一化（勾选框）
        self.norm = tk.BooleanVar()
        self.norm.set(True)
        self.CheckButton1 = tk.Checkbutton(
            self, text='数据集归一化', variable=self.norm)
        self.CheckButton1.grid(row=13, column=4, padx=5, pady=5)
        # 数据集旋转增广（勾选框）
        self.rotate = tk.BooleanVar()
        self.rotate.set(False)
        self.CheckButton2 = tk.Checkbutton(
            self, text='数据集旋转增广', variable=self.rotate)
        self.CheckButton2.grid(row=13, column=5, padx=5, pady=5)
        # 模型输出路径（按钮）
        self.Label2 = tk.Label(self, text='模型输出路径')
        self.Label2.grid(row=14, column=1, padx=5, pady=5)
        self.OutputDir = tk.StringVar()
        # 预设
        # self.OutputDir.set('model')
        self.Entry2 = tk.Entry(self, textvariable=self.OutputDir)
        self.Entry2.grid(row=14, column=2, padx=5, pady=5)
        self.Button2 = tk.Button(
            self, text='选择模型输出路径', command=self.AskOutputDir)
        self.Button2.grid(row=14, column=3, padx=5, pady=5)

        # 生成模型（按钮）
        self.Button3 = tk.Button(self, text='生成模型', command=self.CreateModel)
        self.Button3.grid(row=14, column=4, padx=5, pady=5)

    def ChooseModel(self):
        """选择模型的函数
        在self.RadioButton1,self.RadioButton2,self.RadioButton3中选择一个时触发
        将对应组的组件设置为可用
        剩余组的组件设置为不可用
        同时将self.ModelChoose设置为对应的值
        """
        match(self.ModelChoose.get()):
            case 1:
                self.Entry11['state'] = tk.NORMAL
                self.ComboBox12['state'] = tk.NORMAL
                self.ComboBox13['state'] = tk.NORMAL
                self.Entry14['state'] = tk.NORMAL
                self.Entry15['state'] = tk.NORMAL
                self.ComboBox16['state'] = tk.NORMAL
                self.Entry17['state'] = tk.NORMAL
                self.Entry18['state'] = tk.NORMAL
                self.Entry19['state'] = tk.NORMAL
                self.Entry21['state'] = tk.DISABLED
                self.ComboBox22['state'] = tk.DISABLED
                self.Entry23['state'] = tk.DISABLED
                self.Entry24['state'] = tk.DISABLED
                self.ComboBox25['state'] = tk.DISABLED
                self.Entry26['state'] = tk.DISABLED
                self.Entry27['state'] = tk.DISABLED
                self.ComboBox28['state'] = tk.DISABLED
                self.ComboBox29['state'] = tk.DISABLED
                self.ComboBox210['state'] = tk.DISABLED
                self.ComboBox211['state'] = tk.DISABLED
                self.Entry31['state'] = tk.DISABLED
                self.ComboBox32['state'] = tk.DISABLED
                self.Entry33['state'] = tk.DISABLED
                self.Entry34['state'] = tk.DISABLED
                self.Entry35['state'] = tk.DISABLED
                self.ComboBox36['state'] = tk.DISABLED
                self.Entry37['state'] = tk.DISABLED
                self.Entry38['state'] = tk.DISABLED
                self.ComboBox39['state'] = tk.DISABLED
                self.ComboBox310['state'] = tk.DISABLED
                self.ComboBox311['state'] = tk.DISABLED
                self.ComboBox312['state'] = tk.DISABLED
                self.Entry2['state'] = tk.DISABLED
                self.Button2['state'] = tk.DISABLED
            case 2:
                self.Entry11['state'] = tk.DISABLED
                self.ComboBox12['state'] = tk.DISABLED
                self.ComboBox13['state'] = tk.DISABLED
                self.Entry14['state'] = tk.DISABLED
                self.Entry15['state'] = tk.DISABLED
                self.ComboBox16['state'] = tk.DISABLED
                self.Entry17['state'] = tk.DISABLED
                self.Entry18['state'] = tk.DISABLED
                self.Entry19['state'] = tk.DISABLED
                self.Entry21['state'] = tk.NORMAL
                self.ComboBox22['state'] = tk.NORMAL
                self.Entry23['state'] = tk.NORMAL
                self.Entry24['state'] = tk.NORMAL
                self.ComboBox25['state'] = tk.NORMAL
                self.Entry26['state'] = tk.NORMAL
                self.Entry27['state'] = tk.NORMAL
                self.ComboBox28['state'] = tk.NORMAL
                self.ComboBox29['state'] = tk.NORMAL
                self.ComboBox210['state'] = tk.NORMAL
                self.ComboBox211['state'] = tk.NORMAL
                self.Entry31['state'] = tk.DISABLED
                self.ComboBox32['state'] = tk.DISABLED
                self.Entry33['state'] = tk.DISABLED
                self.Entry34['state'] = tk.DISABLED
                self.Entry35['state'] = tk.DISABLED
                self.ComboBox36['state'] = tk.DISABLED
                self.Entry37['state'] = tk.DISABLED
                self.Entry38['state'] = tk.DISABLED
                self.ComboBox39['state'] = tk.DISABLED
                self.ComboBox310['state'] = tk.DISABLED
                self.ComboBox311['state'] = tk.DISABLED
                self.ComboBox312['state'] = tk.DISABLED
                self.Entry2['state'] = tk.NORMAL
                self.Button2['state'] = tk.NORMAL
            case 3:
                self.Entry11['state'] = tk.DISABLED
                self.ComboBox12['state'] = tk.DISABLED
                self.ComboBox13['state'] = tk.DISABLED
                self.Entry14['state'] = tk.DISABLED
                self.Entry15['state'] = tk.DISABLED
                self.ComboBox16['state'] = tk.DISABLED
                self.Entry17['state'] = tk.DISABLED
                self.Entry18['state'] = tk.DISABLED
                self.Entry19['state'] = tk.DISABLED
                self.Entry21['state'] = tk.DISABLED
                self.ComboBox22['state'] = tk.DISABLED
                self.Entry23['state'] = tk.DISABLED
                self.Entry24['state'] = tk.DISABLED
                self.ComboBox25['state'] = tk.DISABLED
                self.Entry26['state'] = tk.DISABLED
                self.Entry27['state'] = tk.DISABLED
                self.ComboBox28['state'] = tk.DISABLED
                self.ComboBox29['state'] = tk.DISABLED
                self.ComboBox210['state'] = tk.DISABLED
                self.ComboBox211['state'] = tk.DISABLED
                self.Entry31['state'] = tk.NORMAL
                self.ComboBox32['state'] = tk.NORMAL
                self.Entry33['state'] = tk.NORMAL
                self.Entry34['state'] = tk.NORMAL
                self.Entry35['state'] = tk.NORMAL
                self.ComboBox36['state'] = tk.NORMAL
                self.Entry37['state'] = tk.NORMAL
                self.Entry38['state'] = tk.NORMAL
                self.ComboBox39['state'] = tk.NORMAL
                self.ComboBox310['state'] = tk.NORMAL
                self.ComboBox311['state'] = tk.NORMAL
                self.ComboBox312['state'] = tk.NORMAL
                self.Entry2['state'] = tk.NORMAL
                self.Button2['state'] = tk.NORMAL

    def AskDataset(self):
        """选择数据集目录
        选择数据集的函数
        """
        self.DatasetDir.set(filedialog.askopenfilename())

    def AskOutputDir(self):
        """选择模型输出路径
        选择模型输出路径的函数
        """
        self.OutputDir.set(filedialog.askdirectory())

    def CreateModel(self):
        """创建模型的函数
        根据选择的参数，创建MachineLearnModel类或DeepLearningModel类的实例
        """
        match(self.ModelChoose.get()):
            case 1:
                try:
                    batch_size = int(self.value19.get())
                except:
                    batch_size = 'auto'
                try:
                    hidden_layers=tuple(map(int,self.value11.get().split()))
                except:
                    hidden_layers=self.value11.get()
                train_dataset = CifarSklearnDataset(
                    self.DatasetDir.get(), self.norm.get(), self.rotate.get(), 'train')
                val_dataset = CifarSklearnDataset(
                    self.DatasetDir.get(), self.norm.get(), self.rotate.get(), 'val')
                test_dataset = CifarSklearnDataset(
                    self.DatasetDir.get(), self.norm.get(), self.rotate.get(), 'test')
                try:
                    self.model = MachineLearningModel(hidden_layer_sizes=hidden_layers,num_classes=7, activation=self.value12.get(), solver=self.value13.get(), alpha=self.value14.get(), learning_rate_init=self.value15.get(), learning_rate=self.value16.get(), max_iter=self.value17.get(), early_stopping=self.value18.get(), batch_size=self.value19.get(), train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)
                except:
                    messagebox.showerror('错误', '模型参数输入有误')
                    return
            case 2:
                # 模型输出路径
                if(self.OutputDir.get() == ''):
                    messagebox.showerror('错误', '模型输出路径不能为空')
                    return
                # 每层神经元数量
                try:
                    hidden_layers = list(map(int, self.value21.get().split()))
                except:
                    messagebox.showerror('错误', '每层神经元数量输入格式错误，将使用默认值')
                    hidden_layers = [512, 128, 512, 128]
                # 激活函数
                match(self.value22.get()):
                    case 'ReLU':
                        activation = nn.ReLU()
                    case 'Leaky ReLU':
                        activation = nn.LeakyReLU()
                    case 'ELU':
                        activation = nn.ELU()
                    case 'SELU':
                        activation = nn.SELU()
                    case 'Sigmoid':
                        activation = nn.Sigmoid()
                    case 'Tanh':
                        activation = nn.Tanh()
                    case 'Softmax':
                        activation = nn.Softmax()
                    case 'Softplus':
                        activation = nn.Softplus()
                    case 'Hardtanh':
                        activation = nn.Hardtanh()
                # 批大小
                batch_size = self.value23.get()
                # 训练轮数
                epochs = self.value24.get()
                # 是否打乱数据
                shuffle = self.value25.get()
                # 验证频率
                validate_frequency = self.value26.get()
                # 初始学习率
                lr = self.value27.get()
                # 学习率变化
                lr_decent = self.value28.get()
                # 训练设备
                device = self.value29.get()
                # 损失函数
                match(self.value210.get()):
                    case 'CrossEntropyLoss':
                        criterion = nn.CrossEntropyLoss()
                    case 'MSELoss':
                        criterion = nn.MSELoss()
                    case 'BCELoss':
                        criterion = nn.BCELoss()
                    case 'BCEWithLogitsLoss':
                        criterion = nn.BCEWithLogitsLoss()
                    case 'TripletMarginLoss':
                        criterion = nn.TripletMarginLoss()
                    case 'KLDivLoss':
                        criterion = nn.KLDivLoss()
                    case 'CosineEmbeddingLoss':
                        criterion = nn.CosineEmbeddingLoss()
                    case 'MarginRankingLoss':
                        criterion = nn.MarginRankingLoss()
                    case 'PoissonNLLLoss':
                        criterion = nn.PoissonNLLLoss()
                    case 'SoftMarginLoss':
                        criterion = nn.SoftMarginLoss()
                # 优化器
                optimizer = self.value211.get()
                train_dataset = CifarTorchDataset(
                    self.DatasetDir.get(), self.norm.get(), self.rotate.get(), 'train')
                val_dataset = CifarTorchDataset(
                    self.DatasetDir.get(), self.norm.get(), self.rotate.get(), 'val')
                test_dataset = CifarTorchDataset(
                    self.DatasetDir.get(), self.norm.get(), self.rotate.get(), 'test')
                # 模型输出路径
                if(self.OutputDir.get() == ''):
                    messagebox.showerror('错误', '模型输出路径不能为空')
                    return
                try:
                    self.model = DeepLearningModel(net=MLP(num_classes=7, linear_num=hidden_layers, activation=activation),num_classes=7, batch_size=batch_size, epoch=epochs, shuffle=shuffle, Validate_frequency=validate_frequency, lr=lr,
                                          lr_decent=lr_decent, device=device, criterion=criterion, optimizer=optimizer, OutputDir=self.OutputDir.get(), train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)
                except:
                    messagebox.showerror('错误', '模型参数输入有误')
                    return
            case 3:
                # 模型输出路径
                if(self.OutputDir.get() == ''):
                    messagebox.showerror('错误', '模型输出路径不能为空')
                    return
                # 每层神经元数量
                hidden_layers = self.value31.get()
                # 激活函数
                match(self.value32.get()):
                    case 'ReLU':
                        activation = nn.ReLU()
                    case 'Leaky ReLU':
                        activation = nn.LeakyReLU()
                    case 'ELU':
                        activation = nn.ELU()
                    case 'SELU':
                        activation = nn.SELU()
                    case 'Sigmoid':
                        activation = nn.Sigmoid()
                    case 'Tanh':
                        activation = nn.Tanh()
                    case 'Softmax':
                        activation = nn.Softmax()
                    case 'Softplus':
                        activation = nn.Softplus()
                    case 'Hardtanh':
                        activation = nn.Hardtanh()
                # Dropout概率
                dropout = self.value33.get()
                # 批大小
                batch_size = self.value34.get()
                # 训练轮数
                epochs = self.value35.get()
                # 是否打乱数据
                shuffle = self.value36.get()
                # 验证频率
                validate_frequency = self.value37.get()
                # 初始学习率
                lr = self.value38.get()
                # 学习率变化
                lr_decent = self.value39.get()
                # 训练设备
                device = self.value310.get()
                # 损失函数
                match(self.value311.get()):
                    case 'CrossEntropyLoss':
                        criterion = nn.CrossEntropyLoss()
                    case 'MSELoss':
                        criterion = nn.MSELoss()
                    case 'BCELoss':
                        criterion = nn.BCELoss()
                    case 'BCEWithLogitsLoss':
                        criterion = nn.BCEWithLogitsLoss()
                    case 'TripletMarginLoss':
                        criterion = nn.TripletMarginLoss()
                    case 'KLDivLoss':
                        criterion = nn.KLDivLoss()
                    case 'CosineEmbeddingLoss':
                        criterion = nn.CosineEmbeddingLoss()
                    case 'MarginRankingLoss':
                        criterion = nn.MarginRankingLoss()
                    case 'PoissonNLLLoss':
                        criterion = nn.PoissonNLLLoss()
                    case 'SoftMarginLoss':
                        criterion = nn.SoftMarginLoss()
                # 优化器
                optimizer = self.value312.get()
                train_dataset = CifarTorchDataset(
                    self.DatasetDir.get(), self.norm.get(), self.rotate.get(), 'train')
                val_dataset = CifarTorchDataset(
                    self.DatasetDir.get(), self.norm.get(), self.rotate.get(), 'val')
                test_dataset = CifarTorchDataset(
                    self.DatasetDir.get(), self.norm.get(), self.rotate.get(), 'test')
                try:
                    self.model = DeepLearningModel(net=AlexNet(num_classes=7, linear_num=hidden_layers, activation=activation, p=dropout),num_classes=7, batch_size=batch_size, epoch=epochs, shuffle=shuffle, Validate_frequency=validate_frequency,
                                          lr=lr, lr_decent=lr_decent, device=device, criterion=criterion, optimizer=optimizer, OutputDir=self.OutputDir.get(), train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)
                except:
                    messagebox.showerror('错误', '模型参数输入有误')
                    return
        messagebox.showinfo('提示', '模型生成成功')
        self.TrainFrame = tk.Toplevel(self)
        # 绑定窗口关闭事件
        self.TrainFrame.protocol("WM_DELETE_WINDOW", self.on_close)
        ModelTrainUI(self.TrainFrame, self.model)
        self.TrainFrame.mainloop()

    def on_close(self):
        """关闭窗口的函数
        """
        if(isinstance(self.model, DeepLearningModel)):
            self.model.Restart = True
            self.model.Pause = True
        self.TrainFrame.destroy()
        self.model = None
