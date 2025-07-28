from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# 生成数据
np.random.seed(0)
X = np.concatenate([np.random.normal(loc=1, scale=0.5, size=(50, 1)),
                    np.random.normal(loc=3, scale=0.5, size=(50, 1))])
 
# 初始化GaussianMixture模型
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
 
# 使用EM算法拟合数据
gmm.fit(X)
 
# 输出各组成分的权重和均值方差
print("权重s:", gmm.weights_)
print("均值:", gmm.means_)
print("方差:", gmm.covariances_)

# 绘图
fig, ax = plt.subplots(1,2,figsize=(8,4))
sns.distplot(X, ax=ax[0])
ax[0].set_title("数据分布")
sns.distplot(gmm.sample(100)[0], ax=ax[1])
ax[1].set_title("GMM拟合的数据分布")
plt.show()