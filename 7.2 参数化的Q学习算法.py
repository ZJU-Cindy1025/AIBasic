import numpy as np
from scipy import optimize
# 环境参数
GRID_SIZE = 3
ACTIONS = [(0, 1), (1, 0)]  # 向右和向上
START_STATE = (0, 0)  # s1
GOAL_STATE = (2, 2)   # s9
REWARD_GOAL = 1
REWARD_OUT_OF_BOUNDS = -1
REWARD_DEFAULT = 0
EPSILON = 0.1  # 探索率
ALPHA = 0.5    # 学习率
GAMMA = 0.99    # 折扣因子
EPISODES = 1000  # 训练轮次
EITA = 0.1  # 梯度下降比率
# 状态转移函数


def P(state, action):
    next_state = (state[0] + action[0], state[1] + action[1])
    if 0 <= next_state[0] < GRID_SIZE and 0 <= next_state[1] < GRID_SIZE:
        return next_state
    elif (next_state == GOAL_STATE):
        return GOAL_STATE
    else:
        return None

# 奖励函数


def R(state, next_state):
    if next_state == GOAL_STATE:
        return REWARD_GOAL
    elif next_state is None:
        return REWARD_OUT_OF_BOUNDS
    else:
        return REWARD_DEFAULT

# Q学习算法


def DeepQLearning():
    """Q学习算法
    输出:
        策略pi
    """
    # 随机初始化Q_pi在(-1,1)之间
    Q_pi = np.random.uniform(-1, 1, (GRID_SIZE, GRID_SIZE, len(ACTIONS)))
    # 随机初始化参数theta
    theta = np.random.uniform(-1, 1, (GRID_SIZE, GRID_SIZE, len(ACTIONS)))
    # 迭代更新Q_pi直到收敛
    while (True):
        Q_pi_new = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
        # 将s设置为起始状态
        s = START_STATE
        # 重复直到s为终止状态
        while (s != GOAL_STATE and s != None):
            # 以theta为参数执行动作a，观察奖励r和下一状态s'
            if (np.random.uniform(0, 1) < EPSILON):
                a = np.random.choice(range(len(ACTIONS)))
            else:
                a = np.argmax(theta[s[0], s[1]])
            # 执行动作a
            next_s = P(s, ACTIONS[a])
            # 更新参数theta，使用梯度下降法

            def L(theta):
                """L(theta)
                1/2*(R+gamma*max(Q_pi(s',a';theta))-Q_pi(s,a;theta))^2
                """
                if (next_s != None):
                    return 0.5*(R(s, next_s) + GAMMA*np.max(Q_pi[next_s[0], next_s[1]]) - Q_pi[s[0], s[1], a])**2
                else:
                    return 0.5*(R(s, next_s) - Q_pi[s[0], s[1], a])**2
            # 求解L(theta)的梯度
            gradient = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    for k in range(len(ACTIONS)):
                        gradient[i, j, k] = optimize.approx_fprime(
                            np.array([theta[i, j, k]]), L, EITA)[0]
            # 使用梯度下降法更新参数theta
            theta -= EITA * gradient
            # 更新Q值
            Q_pi_new[s[0], s[1], a] = theta[s[0], s[1], a]
            s = next_s
        # 判断是否收敛
        if (np.sum(np.abs(Q_pi_new - Q_pi)) < EPSILON):
            break
        Q_pi = Q_pi_new
    # 计算最优策略
    pi = np.array([[ACTIONS[np.random.choice(len(ACTIONS))]
                    for i in range(GRID_SIZE)] for j in range(GRID_SIZE)])
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            s = (i, j)
            pi[s[0], s[1]] = ACTIONS[np.argmax(Q_pi[s[0], s[1]])]
    return pi


# 测试
pi = DeepQLearning()
for row in pi[::-1]:
    for col in row:
        if (np.all(col == (0, 1))):
            print('→', end=' ')
        else:
            print('↑', end=' ')
    print()
