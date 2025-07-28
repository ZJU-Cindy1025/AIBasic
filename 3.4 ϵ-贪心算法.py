import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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
plt.figure(figsize=(18, 6))

# 第一个子图：每个赌博机收益分布的箱型图
plt.subplot(1, 3, 1)
plt.boxplot([np.random.normal(mu[i], sigma, T) for i in range(K)],
            labels=['赌博机 1', '赌博机 2', '赌博机 3', '赌博机 4', '赌博机 5'])
plt.title('每个赌博机收益分布情况')
plt.xlabel('赌博机臂')
plt.ylabel('收益分布')

# 第二个子图：每个赌博机被选择次数的柱状图
plt.subplot(1, 3, 2)
plt.bar(range(1, K+1), action_counts)
plt.title('每个赌博机被选择的次数')
plt.xlabel('赌博机臂')
plt.ylabel('选择次数')
plt.xticks(range(1, K+1), ['赌博机 1', '赌博机 2', '赌博机 3', '赌博机 4', '赌博机 5'])

# 第三个子图：每个赌博机的平均收益柱状图
average_rewards = [action_rewards[i] / action_counts[i]
                   if action_counts[i] > 0 else 0 for i in range(K)]
plt.subplot(1, 3, 3)
plt.bar(range(1, K+1), average_rewards)
plt.title('每个赌博机的平均收益')
plt.xlabel('赌博机臂')
plt.ylabel('平均收益')
plt.xticks(range(1, K+1), ['赌博机 1', '赌博机 2', '赌博机 3', '赌博机 4', '赌博机 5'])

plt.tight_layout()
plt.show()
