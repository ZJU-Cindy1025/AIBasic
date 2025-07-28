import numpy as np

# 环境参数
GRID_SIZE = 3
ACTIONS = [(0, 1), (1, 0)]  # 向右和向上
START_STATE = (0, 0)  # s1
GOAL_STATE = (2, 2)   # s9
REWARD_GOAL = 1
REWARD_OUT_OF_BOUNDS = -1
REWARD_DEFAULT = 0
ALPHA = 0.5    # 学习率
GAMMA = 0.99    # 折扣因子
EPISODES = 1000  # 训练轮次

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

# 价值迭代算法


def ValueIteration():
    """价值迭代算法
    输出:
        策略pi
    """
    # 随机初始化价值函数
    V_pi = np.zeros((GRID_SIZE, GRID_SIZE))
    # 迭代更新价值函数直到收敛
    while True:
        V_pi_new = np.zeros((GRID_SIZE, GRID_SIZE))
        # 遍历所有状态
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                s = (i, j)
                # 计算当前状态的最优动作
                Q = []
                for action in ACTIONS:
                    next_state = P(s, action)
                    if (next_state != None):
                        q = R(s, next_state) + GAMMA * \
                            V_pi[next_state[0], next_state[1]]
                    else:
                        q = R(s, next_state)
                    Q.append(q)
                V_pi_new[s[0], s[1]] = np.max(Q)
        # 判断是否收敛
        if np.sum(np.abs(V_pi - V_pi_new)) < 1e-5:
            break
        # print(V_pi_new)
        V_pi = V_pi_new.copy()
    # 计算最优策略
    pi = np.array([[ACTIONS[np.random.choice(len(ACTIONS))]
                    for i in range(GRID_SIZE)] for j in range(GRID_SIZE)])
    # 根据价值函数计算最优策略
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            s = (i, j)
            Q = []
            for action in ACTIONS:
                next_state = P(s, action)
                if (next_state != None):
                    q = R(s, next_state) + GAMMA * \
                        V_pi[next_state[0], next_state[1]]
                else:
                    q = R(s, next_state)
                Q.append(q)
            pi[s[0], s[1]] = ACTIONS[np.argmax(Q)]
    return pi


# 测试
pi = ValueIteration()
for row in pi[::-1]:
    for col in row:
        if (np.all(col == (0, 1))):
            print('→', end=' ')
        else:
            print('↑', end=' ')
    print()
