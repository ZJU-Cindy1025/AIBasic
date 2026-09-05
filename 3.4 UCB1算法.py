import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

# 参数设定
K = 5  # 赌博机数量
mu = [0.3, 0.4, 0.5, 0.6, 0.7]  # 分布均值
sigma = np.sqrt(1/75)  # 标准差

# 迭代次数
T = 10000

# 初始化存储结果的数组
action_counts = np.zeros(K)  # 每台赌博机被选择的次数
action_rewards = np.zeros(K)  # 每台赌博机的累计收益

# 拉动赌博机臂的函数


def pull_arm(action):
    reward = np.random.normal(mu[action], sigma)
    return reward

# 上限置信区间算法UCB1


def UCB1():
    # 初始化动作值估计
    Q = np.zeros(K)
    N = np.zeros(K)

    for t in range(T):
        # UCB1算法中的动作选择
        if np.any(N == 0):
            # 探索：选择未被选择过的动作
            action = np.where(N == 0)[0][0]
        else:
            # 利用：选择上限置信区间最高的动作
            action = np.argmax(Q + np.sqrt(2 * np.log(t) / N))

        # 更新动作计数
        action_counts[action] += 1

        # 拉动选择的赌博机臂
        reward = pull_arm(action)

        # 更新动作值估计（样本平均法）
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]

        # 更新每台赌博机的累计收益
        action_rewards[action] += reward


# 运行UCB1算法
rewards = UCB1()

# 绘制到同一行的figure中
fig = make_subplots(rows=1, cols=3, subplot_titles=[
                    '每个赌博机收益分布情况', '每个赌博机被选择的次数', '每个赌博机的平均收益'])

# 第一个子图：每个赌博机收益分布的箱型图
for i in range(K):
    fig.add_trace(px.box(y=np.random.normal(mu[i], sigma, T),
                         labels={'y': f'赌博机 {i + 1}'}).data[0], row=1, col=1)

# 第二个子图：每个赌博机被选择次数的柱状图
fig.add_trace(px.bar(x=list(range(1, K + 1)),
              y=action_counts).data[0], row=1, col=2)

# 第三个子图：每个赌博机的平均收益柱状图
average_rewards = [action_rewards[i] / action_counts[i]
                   if action_counts[i] > 0 else 0 for i in range(K)]
fig.add_trace(px.bar(x=list(range(1, K + 1)),
              y=average_rewards).data[0], row=1, col=3)
fig.show()
