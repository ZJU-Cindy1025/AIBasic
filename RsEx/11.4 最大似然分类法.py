import skimage as ski
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取图像
image = ski.data.coins()
# 转换训练标签
train_label = ski.io.imread('RsEx/coins_KNN_MLE/label.png')
train_label = train_label[:, :, 0]
# 转换真实标签
true_label = ski.io.imread('RsEx/coins_label/label.png')
true_label = true_label[:, :, 0]

# 最大似然分类法
clf = GaussianNB()
clf.fit(image.reshape(-1, 1), train_label.reshape(-1))

# 应用训练模型
result_label = clf.predict(image.reshape(-1, 1)).reshape(image.shape)

# 显示分类结果
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(image, cmap='gray')
ax[0, 0].set_title('原图')
ax[0, 0].axis('off')
ax[0, 1].imshow(train_label, cmap='gray')
ax[0, 1].set_title('训练标签')
ax[0, 1].axis('off')
ax[1, 0].imshow(true_label, cmap='gray')
ax[1, 0].set_title('真实标签')
ax[1, 0].axis('off')
ax[1, 1].imshow(result_label, cmap='gray')
ax[1, 1].set_title('最大似然分类标签')
ax[1, 1].axis('off')
plt.show()
plt.imsave('RsEx/coins_result/coins_MLE_result.png', result_label, cmap="gray")

# 计算混淆矩阵
matrix = confusion_matrix(true_label.reshape(-1), result_label.reshape(-1))
print('混淆矩阵:')
print(matrix)
print('查准率:', matrix[1, 1]/(matrix[1, 1]+matrix[0, 1]))
print('查全率:', matrix[1, 1]/(matrix[1, 1]+matrix[1, 0]))
