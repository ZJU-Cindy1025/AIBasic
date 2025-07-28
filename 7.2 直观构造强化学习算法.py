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

# 基于动态规划的策略评估


def DPPolicyEvaluation(pi):
    """基于动态规划的策略评估
    输入:
        pi: 策略
        P: 状态转移函数
        R: 奖励函数
    输出:
        V_pi: 价值函数
    """
    # 随机初始化价值函数
    V_pi = np.random.uniform(-1, 1, (GRID_SIZE, GRID_SIZE))
    V_pi_new = np.zeros((GRID_SIZE, GRID_SIZE))
    # 迭代更新价值函数直到收敛
    while True:
        # 遍历所有状态
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                # 当前状态
                s = (i, j)
                # 计算当前状态的价值
                next_state = P(s, pi[s[0], s[1]])
                if (next_state != None):
                    V_pi_new[s[0], s[1]] = R(
                        s, next_state) + GAMMA * V_pi[next_state[0], next_state[1]]
                else:
                    V_pi_new[s[0], s[1]] = R(s, next_state)
        # 判断是否收敛
        if np.sum(np.abs(V_pi_new - V_pi)) < 1e-6:
            break
        V_pi = V_pi_new.copy()
    return V_pi

# 基于蒙特卡洛方法的策略评估


def MCPolicyEvaluation(pi):
    """基于蒙特卡洛方法的策略评估
    输入:
        pi: 策略
    输出:
        V_pi: 价值函数
    """
    # 随机初始化V_pi
    V_pi = np.random.uniform(-1, 1, (GRID_SIZE, GRID_SIZE))
    # 对任意s，初始化s.returns为空列表
    returns = {(i, j): [] for i in range(GRID_SIZE) for j in range(GRID_SIZE)}
    # 重复迭代若干次
    for _ in range(EPISODES):
        # 根据pi产生片段D
        D = []
        s = START_STATE
        while s != GOAL_STATE and s != None:
            action = pi[s[0], s[1]]
            next_state = P(s, action)
            reward = R(s, next_state)
            D.append((s, action, reward))
            s = next_state
        # 计算G，s在D中第一次出现时的反馈
        G = 0
        for i, (s, action, reward) in enumerate(D[::-1]):
            G = GAMMA * G + reward
            if s not in [x[0] for x in D[:-i]]:
                # 将G添加到s.returns中
                returns[s].append(G)
                # 更新价值函数
                V_pi[s[0], s[1]] = np.mean(returns[s])
    return V_pi

# 基于时序差分法的策略评估


def TemporalDifference(pi):
    """基于时序差分法的策略评估

    输入:
        pi: 策略
    输出:
        V_pi: 价值函数
    """
    # 随机初始化V_pi
    V_pi = np.random.uniform(-1, 1, (GRID_SIZE, GRID_SIZE))
    V_pi_new = np.zeros((GRID_SIZE, GRID_SIZE))
    # 迭代更新V_pi直到收敛
    while (True):
        # 将s初始化为起始状态
        s = START_STATE
        # 重复直到s为终止状态
        while s != GOAL_STATE and s != None:
            # 选择动作
            action = pi[s[0], s[1]]
            # 执行动作a，观察奖励R和下一个状态s'
            next_state = P(s, action)
            reward = R(s, next_state)
            # 更新价值函数
            if (next_state != None):
                V_pi_new[s[0], s[1]] += ALPHA * (
                    reward + GAMMA * V_pi_new[next_state[0], next_state[1]] - V_pi_new[s[0], s[1]])
            else:
                V_pi_new[s[0], s[1]] += ALPHA * \
                    (reward - V_pi_new[s[0], s[1]])
            s = next_state
        # 判断是否收敛
        if np.sum(np.abs(V_pi_new - V_pi)) < 1e-5:
            break
        V_pi = V_pi_new.copy()
    return V_pi

# 直观构造强化学习算法


def PolicyIteration(PolicyEvaluation):
    """直观构造强化学习算法
    输入:
        马尔可夫决策过程
    输出:
        pi: 最优策略
    """
    # 随机初始化策略
    pi = np.array([[ACTIONS[np.random.choice(len(ACTIONS))]
                    for i in range(GRID_SIZE)] for j in range(GRID_SIZE)])
    while True:
        # 策略评估：上面方法三选一
        q_pi = PolicyEvaluation(pi)
        # 遍历每一个状态
        policy_stable = True
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                s = (i, j)
                old_action = pi[s[0], s[1]]
                # 更新策略
                Q = []
                for action in ACTIONS:
                    next_state = P(s, action)
                    if (next_state != None):
                        q = R(s, next_state) + GAMMA * \
                            q_pi[next_state[0], next_state[1]]
                    else:
                        q = R(s, next_state)
                    Q.append(q)
                pi[s[0], s[1]] = ACTIONS[np.argmax(Q)]
                if np.all(old_action != pi[s[0], s[1]]):
                    policy_stable = False
        if policy_stable:
            break
    return pi


# 测试
pi = PolicyIteration(DPPolicyEvaluation)
print('基于动态规划的策略评估结果')
for row in pi[::-1]:
    for col in row:
        if (np.all(col == (0, 1))):
            print('→', end=' ')
        else:
            print('↑', end=' ')
    print()
print()
pi = PolicyIteration(MCPolicyEvaluation)
print('基于蒙特卡洛的策略评估结果')
for row in pi[::-1]:
    for col in row:
        if (np.all(col == (0, 1))):
            print('→', end=' ')
        else:
            print('↑', end=' ')
    print()
print()
pi = PolicyIteration(TemporalDifference)
print('基于时序差分的策略评估结果')
for row in pi[::-1]:
    for col in row:
        if (np.all(col == (0, 1))):
            print('→', end=' ')
        else:
            print('↑', end=' ')
    print()
