import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def TF_MLP():
    #(1)导入必要的库
    #(2)准备数据
    #这里以随机生成的数据为例，展示如何准备数据
    np.random.seed(0)
    x_train = np.random.random((1000, 20))  # 1000个样本，每个样本20个特征
    y_train = np.random.randint(2, size=(1000, 1))  # 1000个标签，0或1

    #(3)定义模型结构
    model = Sequential([
        # 输入层-隐含层1，20个输入特征，64个神经元，ReLU激活函数
        Dense(64, activation='relu', input_shape=(20,)),
        # 隐藏层2，32个神经元，ReLU激活函数
        Dense(32, activation='relu'),
        # 隐藏层3，16个神经元，ReLU激活函数
        Dense(16, activation='relu'),
        # 隐藏层4，8个神经元，ReLU激活函数
        Dense(8, activation='relu'),
        # 输出层，1个神经元，sigmoid激活函数用于二分类
        Dense(1, activation='sigmoid')
    ])

    #（4）编译模型
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    #（5）训练模型
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    #（6）评估模型
    x_test,y_test=x_train, y_train
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score)
    print('Test accuracy:', acc)

    #（7）使用模型进行预测
    x_new=np.random.random((1, 20))
    prediction = model.predict(x_new)
    print(prediction)
    
TF_MLP()