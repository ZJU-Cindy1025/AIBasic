import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

# 参数设定
K = 5  # 赌博机数量
mu = [0.3, 0.4, 0.5, 0.6, 0.7]  # 分布均值
sigma = np.sqrt(1/75)  # 标准差

# ϵ-贪心算法中的ϵ设定
epsilon = 0.1

# 迭代次数
T = 10000

# 初始化存储结果的数组
action_counts = np.zeros(K)  # 每台赌博机被选择的次数
action_rewards = np.zeros(K)  # 每台赌博机的累计收益

# 拉动赌博机臂的函数


def pull_arm(action):
    reward = np.random.normal(mu[action], sigma)
    return reward

# ϵ-贪心算法


def epsilon_greedy(epsilon):
    # 初始化动作值估计
    Q = np.array([np.inf for i in range(K)])  # 未摇动的赌博机的期望收益为无穷大
    N = np.zeros(K)

    for t in range(T):
        # ϵ-贪心算法中的动作选择
        if np.random.rand() < epsilon:
            # 探索：随机选择一个动作
            action = np.random.randint(K)
        else:
            # 利用：选择动作值估计最高的动作
            action = np.argmax(Q)

        # 更新动作计数
        action_counts[action] += 1

        # 拉动选择的赌博机臂
        reward = pull_arm(action)

        # 更新动作值估计（样本平均法）
        N[action] += 1
        if (N[action] == 1):
            Q[action] = reward
        else:
            Q[action] = (Q[action]*(N[action]-1)+reward)/N[action]

        # 更新每台赌博机的累计收益
        action_rewards[action] += reward


# 运行ϵ-贪心算法
rewards = epsilon_greedy(epsilon)

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
