import unittest
from ModelChooseUI import ModelChooseUI
from ModelTrainUI import ModelTrainUI
import tkinter as tk


class TestUI(unittest.TestCase):
    def test_ModelChooseUI(self):
        root = tk.Tk()
        root.title('AI综合实践-模型选择窗口测试')
        app = ModelChooseUI(root)
        root.mainloop()

    def test_ModelTrainUI(self):
        root = tk.Tk()
        root.title('AI综合实践-模型训练窗口测试')
        app = ModelTrainUI(root, model=None)
        root.mainloop()


if __name__ == '__main__':
    unittest.main()
