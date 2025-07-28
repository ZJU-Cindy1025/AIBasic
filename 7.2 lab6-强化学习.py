import numpy as np

# 环境参数
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 向右，向上，向左，向下
START_STATE = (0, 0)  # 起始状态
GOAL_STATE = (3, 4)  # 获胜终局状态
FAIL_STATE = [(1, 1), (1, 4), (2, 2)]  # 失败终局状态
ALPHA = 0.5    # 学习率
GAMMA = 0.99    # 折扣因子
EPISODES = 1000  # 训练轮次
EPSILON = 0.1  # 探索率
# 奖励表
REWARD_TABLE = np.array([
    [0, 0, 0, 0, 0],
    [0, -1, 0, 0, -1],
    [0, 0, -1, 0, 0],
    [0, 0, 0, 0, 1]
])

# 状态转移函数


def P(state, action):
    next_state = (state[0] + action[0], state[1] + action[1])
    return next_state

# 奖励函数


def R(state, next_state):
    return REWARD_TABLE[next_state[0], next_state[1]]

# Q学习算法


def QLearning():
    # 随机初始化Q_pi
    Q_pi = np.full((4, 5, len(ACTIONS)), [0.2, 0.2, 0, 0])
    Q_pi_new = np.full((4, 5, len(ACTIONS)), [0.2, 0.2, 0, 0])
    # 迭代更新Q_pi直到收敛
    while (True):
        # 将s设置为起始状态
        s = START_STATE
        # 重复直到s为终止状态
        while (s != GOAL_STATE and s not in FAIL_STATE):
            # 执行动作a，观察奖励r和下一状态s'
            if (np.random.uniform(0, 1) < EPSILON):
                # 注意，执行动作a后，不能超出边界，需要根据s分情况讨论
                if (s[0] == 0 and s[1] == 0):
                    a = np.random.choice(range(2))
                elif (s[0] == 0 and s[1] == 4):
                    a = np.random.choice([1, 2])
                elif (s[0] == 3 and s[1] == 0):
                    a = np.random.choice([0, 3])
                elif (s[0] == 3 and s[1] == 4):
                    a = np.random.choice([2, 3])
                elif (s[0] == 0 and s[1] != 0 and s[1] != 4):
                    a = np.random.choice([0, 1, 2])
                elif (s[0] == 3 and s[1] != 0 and s[1] != 4):
                    a = np.random.choice([0, 2, 3])
                elif (s[1] == 0 and s[0] != 0 and s[0] != 3):
                    a = np.random.choice([0, 3, 1])
                elif (s[1] == 4 and s[0] != 0 and s[0] != 3):
                    a = np.random.choice([1, 2, 3])
                else:
                    a = np.random.choice(range(len(ACTIONS)))
            else:
                # 注意，执行动作a后，不能超出边界，需要根据s分情况讨论
                if (s[0] == 0 and s[1] == 0):
                    slice_indices = range(2)
                    Q_pi_slice = Q_pi[s[0], s[1], slice_indices].copy()
                    max_value = np.max(Q_pi_slice)
                    index_in_slice = np.where(Q_pi_slice == max_value)[0][0]
                    a = slice_indices[index_in_slice]
                elif (s[0] == 0 and s[1] == 4):
                    slice_indices = [1, 2]
                    Q_pi_slice = Q_pi[s[0], s[1], slice_indices].copy()
                    max_value = np.max(Q_pi_slice)
                    index_in_slice = np.where(Q_pi_slice == max_value)[0][0]
                    a = slice_indices[index_in_slice]
                elif (s[0] == 3 and s[1] == 0):
                    slice_indices = [0, 3]
                    Q_pi_slice = Q_pi[s[0], s[1], slice_indices].copy()
                    max_value = np.max(Q_pi_slice)
                    index_in_slice = np.where(Q_pi_slice == max_value)[0][0]
                    a = slice_indices[index_in_slice]
                elif (s[0] == 3 and s[1] == 4):
                    slice_indices = [2, 3]
                    Q_pi_slice = Q_pi[s[0], s[1], slice_indices].copy()
                    max_value = np.max(Q_pi_slice)
                    index_in_slice = np.where(Q_pi_slice == max_value)[0][0]
                    a = slice_indices[index_in_slice]
                elif (s[0] == 0 and s[1] != 0 and s[1] != 4):
                    slice_indices = [0, 1, 2]
                    Q_pi_slice = Q_pi[s[0], s[1], slice_indices].copy()
                    max_value = np.max(Q_pi_slice)
                    index_in_slice = np.where(Q_pi_slice == max_value)[0][0]
                    a = slice_indices[index_in_slice]
                elif (s[0] == 3 and s[1] != 0 and s[1] != 4):
                    slice_indices = [0, 2, 3]
                    Q_pi_slice = Q_pi[s[0], s[1], slice_indices].copy()
                    max_value = np.max(Q_pi_slice)
                    index_in_slice = np.where(Q_pi_slice == max_value)[0][0]
                    a = slice_indices[index_in_slice]
                elif (s[1] == 0 and s[0] != 0 and s[0] != 3):
                    slice_indices = [0, 3, 1]
                    Q_pi_slice = Q_pi[s[0], s[1], slice_indices].copy()
                    max_value = np.max(Q_pi_slice)
                    index_in_slice = np.where(Q_pi_slice == max_value)[0][0]
                    a = slice_indices[index_in_slice]
                elif (s[1] == 4 and s[0] != 0 and s[0] != 3):
                    slice_indices = [1, 2, 3]
                    Q_pi_slice = Q_pi[s[0], s[1], slice_indices].copy()
                    max_value = np.max(Q_pi_slice)
                    index_in_slice = np.where(Q_pi_slice == max_value)[0][0]
                    a = slice_indices[index_in_slice]
                else:
                    a = np.argmax(Q_pi[s[0], s[1]])
            # 执行动作a
            next_s = P(s, ACTIONS[a])
            # 更新Q值
            Q_pi_new[s[0], s[1], a] = Q_pi[s[0], s[1], a] + ALPHA * \
                (R(s, next_s) + GAMMA *
                 np.max(Q_pi[next_s[0], next_s[1]]) - Q_pi[s[0], s[1], a])
            s = next_s
        # 判断是否收敛
        if (np.sum(np.abs(Q_pi_new - Q_pi)) < 1e-6):
            break
        Q_pi = Q_pi_new.copy()
    # 计算最优策略
    pi = np.array([[ACTIONS[np.random.choice(len(ACTIONS))]
                  for i in range(5)] for j in range(4)])
    for i in range(4):
        for j in range(5):
            s = (i, j)
            pi[s[0], s[1]] = ACTIONS[np.argmax(Q_pi[s[0], s[1]])]
    return pi


# 测试
pi = QLearning()
for row in pi:
    for col in row:
        if (np.all(col == (0, 1))):
            print('→', end=' ')
        elif (np.all(col == (1, 0))):
            print('↓', end=' ')
        elif (np.all(col == (0, -1))):
            print('←', end=' ')
        elif (np.all(col == (-1, 0))):
            print('↑', end=' ')
    print()
