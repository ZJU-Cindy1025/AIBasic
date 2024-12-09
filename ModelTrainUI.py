import tkinter as tk
import threading
from Model import MachineLearningModel, DeepLearningModel
from tkinter.simpledialog import askinteger
from tkinter.messagebox import showerror
import os
import torch

class ModelTrainUI(tk.Frame):
    """
    模型训练界面
    """

    def __init__(self, master, model):
        """初始化函数
        model: MachineLearningModel 或 DeepLearningModel，从父窗口传入的模型对象
        """
        super().__init__(master)
        self.pack()
        self.model = model
        self.create_widgets()
        # 每隔10ms检查一次是否有新的日志消息
        self.after(10, self.UpdateLog)

    def create_widgets(self):
        self.scrollbar = tk.Scrollbar(self)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text = tk.Text(self)
        self.text.pack(padx=10, pady=10)
        self.text.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.text.yview)
        self.start_button = tk.Button(
            self, text='开始训练', command=self.StartTraining)
        self.start_button.pack(side='left', padx=10, pady=10)
        self.pause_button = tk.Button(
            self, text='暂停训练', command=self.PauseTraining)
        self.pause_button.pack(side='left', padx=10, pady=10)
        self.resume_button = tk.Button(
            self, text='继续训练', command=self.ResumeTraining)
        self.resume_button.pack(side='left', padx=10, pady=10)
        self.restart_button = tk.Button(
            self, text='重新训练', command=self.RestartTraining)
        self.restart_button.pack(side='left', padx=10, pady=10)
        self.test_button = tk.Button(self, text='测试模型', command=self.TestModel)
        self.test_button.pack(side='left', padx=10, pady=10)
        self.start_button.config(state='normal')
        self.pause_button.config(state='disabled')
        self.resume_button.config(state='disabled')
        self.restart_button.config(state='disabled')
        if (isinstance(self.model, MachineLearningModel)):
            self.test_button.config(state='disabled')
        elif (isinstance(self.model, DeepLearningModel)):
            self.test_button.config(state='normal')

    def StartTraining(self):
        """开始训练"""
        self.model.Restart = False
        self.model.Pause = False
        self.Training_thread = threading.Thread(target=self.model.Train)
        self.Training_thread.start()
        # 设置开始训练按钮不可用
        self.start_button.config(state='disabled')
        # 设置暂停训练按钮可用
        if (isinstance(self.model, DeepLearningModel)):
            self.pause_button.config(state='normal')
        # 设置继续训练按钮不可用
        self.resume_button.config(state='disabled')
        # 设置重新训练按钮不可用
        self.restart_button.config(state='disabled')
        # 设置测试模型按钮不可用
        self.test_button.config(state='disabled')
        # 在self.Training_thread线程结束后，设置按钮状态

        def on_training_end():
            if (hasattr(self, 'Training_thread')):
                self.Training_thread.join()  # 等待训练线程结束
            if (self.model.Restart == False):
                self.restart_button.config(state='normal')
                self.test_button.config(state='normal')
                self.pause_button.config(state='disabled')
                self.resume_button.config(state='disabled')
        # 在训练线程结束时调用 on_training_end 来恢复界面按钮状态
        self.Training_Waiting_thread = threading.Thread(target=on_training_end)
        self.Training_Waiting_thread.start()

    def ResumeTraining(self):
        """继续训练"""
        self.model.Pause = False
        # 设置开始训练按钮不可用
        self.start_button.config(state='disabled')
        # 设置继续训练按钮不可用
        self.resume_button.config(state='disabled')
        # 设置暂停训练按钮可用
        self.pause_button.config(state='normal')
        # 设置重新训练按钮不可用
        self.restart_button.config(state='disabled')
        # 设置测试模型按钮不可用
        self.test_button.config(state='disabled')

    def PauseTraining(self):
        """暂停训练"""
        self.model.Pause = True
        # 设置开始训练按钮不可用
        self.start_button.config(state='disabled')
        # 设置继续训练按钮可用
        self.resume_button.config(state='normal')
        # 设置暂停训练按钮不可用
        self.pause_button.config(state='disabled')
        # 设置重新训练按钮可用
        self.restart_button.config(state='normal')
        # 设置测试模型按钮不可用
        self.test_button.config(state='disabled')

    def RestartTraining(self):
        """重新训练"""
        if (isinstance(self.model, DeepLearningModel)):
            self.model.Restart = True
        # 如果self.Training_thread存在，等待训练线程结束
        if (hasattr(self, 'Training_thread')):
            self.Training_thread.join()
        # 清空日志框
        self.text.delete(1.0, tk.END)
        # 更新UI状态
        self.start_button.config(state='normal')  # 使开始训练按钮可用
        self.pause_button.config(state='disabled')  # 禁用暂停按钮
        self.resume_button.config(state='disabled')  # 禁用继续训练按钮
        self.restart_button.config(state='disabled')  # 禁用重新训练按钮
        self.test_button.config(state='disabled')  # 禁用测试按钮
        if (isinstance(self.model, DeepLearningModel)):
            # 清空self.model.OutputDir下的模型文件
            for file in os.listdir(self.model.OutputDir):
                if file.endswith('.pth'):
                    os.remove(self.model.OutputDir + '/' + file)

    def TestModel(self):
        """测试模型"""
        if (isinstance(self.model, DeepLearningModel)):
            modelnum = askinteger('测试模型', '测试第几个模型？')
            try:
                self.model.net.load_state_dict(torch.load(self.model.OutputDir+'/model_%d.pth' % modelnum))
            except:
                showerror('错误', '模型文件不存在或不匹配')
                return
            self.Test_thread = threading.Thread(
                    target=self.model.Test, args=(modelnum,))
            self.Test_thread.start()
        elif (isinstance(self.model, MachineLearningModel)):
            self.Test_thread = threading.Thread(target=self.model.Test)
            self.Test_thread.start()
        # 设置开始训练按钮不可用
        self.start_button.config(state='disabled')
        # 设置测试模型按钮不可用
        self.test_button.config(state='disabled')
        # 设置暂停训练按钮不可用
        self.pause_button.config(state='disabled')
        # 设置继续训练按钮不可用
        self.resume_button.config(state='disabled')
        # 设置重新训练按钮不可用
        self.restart_button.config(state='disabled')

        def on_testing_end():
            self.Test_thread.join()  # 等待测试线程结束
            # 设置开始训练按钮不可用
            self.start_button.config(state='disabled')
            # 设置测试模型按钮可用
            self.test_button.config(state='normal')
            # 设置暂停训练按钮不可用
            self.pause_button.config(state='disabled')
            # 设置继续训练按钮不可用
            self.resume_button.config(state='disabled')
            # 设置重新训练按钮可用
            self.restart_button.config(state='normal')
        # 在测试线程结束时调用 on_training_end 来恢复界面按钮状态
        self.Test_Waiting_Thread = threading.Thread(target=on_testing_end)
        self.Test_Waiting_Thread.start()

    def UpdateLog(self):
        """更新文本框"""
        while not self.model.log.empty():
            log_message = self.model.log.get()
            self.text.insert(tk.END, log_message + '\n')
            self.text.yview(tk.END)  # 自动滚动到最底部
        self.after(10, self.UpdateLog)
