import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
#加载和预处理数据
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# 查看数据类型 type(train_images)  # numpy.ndarray
print(train_images.shape)

 # 归一化数据
train_images, test_images = train_images / 255.0, test_images / 255.0

# 将标签转换为独热编码
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)
model = models.Sequential() #初始一个前馈神经网络

# 增加卷积运算层1，卷积核数量=32个，卷积核大小3*3，激活函数relu
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# 增加池化运算层1，感受野2*2
model.add(layers.MaxPooling2D((2, 2)))
# 增加卷积运算层2，卷积核数量=64个，卷积核大小3*3，激活函数relu
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 增加池化运算层2，池化窗口（感受野）2*2
model.add(layers.MaxPooling2D((2, 2)))
# 增加卷积运算层3，卷积核数量=64个，卷积核大小3*3，激活函数relu
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 添加全连接层，即MLP
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#编译模型
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#训练模型
history = model.fit(train_images, train_labels, epochs=3,
                    validation_data=(test_images, test_labels))

#评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc) #Test accuracy: 0.7084000110626221
  
acc = history.history['accuracy']  
val_acc = history.history['val_accuracy']  
loss = history.history['loss']  
val_loss = history.history['val_loss']  
  
epochs = range(len(acc))  
  
plt.plot(epochs, acc, 'r', label='Training accuracy')  
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')  
plt.title('Training and validation accuracy')  
plt.legend(loc=0)  
plt.figure()  
  
plt.plot(epochs, loss, 'r', label='Training Loss')  
plt.plot(epochs, val_loss, 'b', label='Validation Loss')  
plt.title('Training and validation loss')  
plt.legend(loc=0)  
  
plt.show()