# 基于MCTS算法和pygame库的黑白棋游戏
import pygame
import math
import random
from tqdm import tqdm

# 棋盘


class Board:
    # 初始化棋盘
    def __init__(self):
        self.board = [['' for _ in range(BOARD_COLS)]
                      for _ in range(BOARD_ROWS)]
        self.board[3][3] = 'X'
        self.board[3][4] = 'O'
        self.board[4][3] = 'O'
        self.board[4][4] = 'X'

    # 绘制棋盘和棋子
    def draw_board(self):
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                pygame.draw.rect(
                    screen, BG_COLOR, (c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                pygame.draw.rect(screen, LINE_COLOR, (c * SQUARE_SIZE,
                                 r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), LINE_WIDTH)
                if self.board[r][c] == 'X':
                    pygame.draw.circle(screen, BLACK, (c * SQUARE_SIZE + SQUARE_SIZE //
                                       2, r * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 5)
                elif self.board[r][c] == 'O':
                    pygame.draw.circle(screen, WHITE, (c * SQUARE_SIZE + SQUARE_SIZE //
                                       2, r * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 5)

    # 落子
    def drop_piece(self, row, col, player):
        self.board[row][col] = player
        self.flip_pieces(row, col, player)

    # 获取合法动作
    def get_legal_actions(self, player):
        actions = []
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if self.is_valid_move(r, c, player):
                    actions.append((r, c))
        return actions

    # 棋子翻转
    def flip_pieces(self, row, col, player):
        if player == 'X':
            opponent = 'O'
        else:
            opponent = 'X'
        directions = [[-1, -1], [-1, 0], [-1, 1],
                      [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        for d in directions:
            x, y = row + d[0], col + d[1]
            if x >= 0 and x < BOARD_ROWS and y >= 0 and y < BOARD_COLS and self.board[x][y] == opponent:
                x, y = x + d[0], y + d[1]
                while x >= 0 and x < BOARD_ROWS and y >= 0 and y < BOARD_COLS and self.board[x][y] == opponent:
                    x, y = x + d[0], y + d[1]
                if x >= 0 and x < BOARD_ROWS and y >= 0 and y < BOARD_COLS and self.board[x][y] == player:
                    x, y = row + d[0], col + d[1]
                    while x >= 0 and x < BOARD_ROWS and y >= 0 and y < BOARD_COLS and self.board[x][y] == opponent:
                        self.board[x][y] = player
                        x, y = x + d[0], y + d[1]

    # 判断是否合法
    def is_valid_move(self, row, col, player):
        if self.board[row][col] != '':
            return False
        if player == 'X':
            opponent = 'O'
        else:
            opponent = 'X'
        directions = [[-1, -1], [-1, 0], [-1, 1],
                      [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        for d in directions:
            x, y = row + d[0], col + d[1]
            if x >= 0 and x < BOARD_ROWS and y >= 0 and y < BOARD_COLS and self.board[x][y] == opponent:
                x, y = x + d[0], y + d[1]
                while x >= 0 and x < BOARD_ROWS and y >= 0 and y < BOARD_COLS and self.board[x][y] == opponent:
                    x, y = x + d[0], y + d[1]
                if x >= 0 and x < BOARD_ROWS and y >= 0 and y < BOARD_COLS and self.board[x][y] == player:
                    return True
        return False

    # 判断是否游戏结束
    def terminal_test(self):
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if self.board[r][c] == '':
                    return False
        return True

    # 计算棋子数量
    def count_pieces(self):
        x_count, o_count = 0, 0
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if self.board[r][c] == 'X':
                    x_count += 1
                elif self.board[r][c] == 'O':
                    o_count += 1
        return x_count, o_count

    # 计算胜负
    def calculate_winner(self):
        x_count, o_count = 0, 0
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if self.board[r][c] == 'X':
                    x_count += 1
                elif self.board[r][c] == 'O':
                    o_count += 1
        if x_count > o_count:
            return 'X'
        elif x_count < o_count:
            return 'O'
        else:
            return 'T'

# 蒙特卡洛树


class Node:
    def __init__(self, state: Board, parent=None):
        # 此时的棋盘状态
        self.state = state
        # 父节点
        self.parent = parent
        # 子节点
        self.children = {}
        # 访问次数
        self.N = 0
        # 每个节点的累计分数
        self.Q = 0
        # 可选动作
        self.action_list = state.get_legal_actions('O')
        # 是否被扩展
        self.is_expanded = False

    # UCB1公式
    def get_value(self, C):
        if self.N == 0:
            return 0
        return self.Q / self.N + C * math.sqrt(math.log(self.parent.N) / self.N)

# 上限置信区间蒙特卡洛树搜索


def UCT_search(s0: Node, max_iter: int):
    """上限置信区间蒙特卡洛树搜索

    输入：当前状态s0
    输出：玩家MAX行动下，当前最优动作a*
    """
    # 创建新的节点v
    v0 = Node(s0.state)
    # 在未达到最大迭代次数时迭代
    for _ in tqdm(range(max_iter)):
        # 选择策略
        vl = SelectPolicy(v0)
        # 模拟
        st = SimulatePolicy(vl.state)
        # 反向传播
        BackPropagate(vl, st)
    # 返回置信上限最大的动作a*
    return UCB1(v0, 0)

# 选择策略


def SelectPolicy(v0: Node):
    """选择策略

    输入：选择的起始节点v0
    输出：选择步骤的结束节点v
    """
    # 一开始将v赋值给v0
    v = v0
    # 当v不是终局状态时
    while not v.state.terminal_test():
        # 如果v有未被扩展的子节点
        if not v.is_expanded:
            # 返回未被扩展的后继结点v'
            return Expand(v)
        # 否则，选择置信上限最大的动作a*
        else:
            v = UCB1(v, 1)
    # 返回v
    return v

# 模拟


def SimulatePolicy(s0: Board):
    """模拟

    输入：状态s0
    输出：模拟的终止状态s
    """
    # 一开始将s赋值给s0
    s = s0
    # 当s不是终局状态时
    while not s.terminal_test():
        # 随机选择一个合法动作a
        if (s.get_legal_actions('O') == []):
            break
        a = random.choice(s.get_legal_actions('O'))
        # 在状态s下执行动作a
        s.drop_piece(a[0], a[1], 'O')
    # 返回终局
    return s

# 反向传播


def BackPropagate(v: Node, st: Board):
    """反向传播

    输入：反向传播更新的起始节点v，终局状态st
    """
    # 当v不是空节点时
    while (v != None):
        # 更新节点v的访问次数N和Q值
        v.N += 1
        # 记录该分数是为了让当前节点的父节点选择一个置信上限最大的子节点，因此要最大化父节点的Q值，即减去当前节点的分数，也就是减去当前节点的参考分数
        v.Q -= st.count_pieces()[1]-st.count_pieces()[0]
        # v指向其父节点
        v = v.parent

# 展开子节点


def Expand(v: Node):
    """展开子节点
    输入：节点v
    输出：未被扩展的后继结点v'
    """
    # 如果v没有合法动作
    if (v.action_list == []):
        return v
    # 随机选择一个未探索的合法动作a
    a = random.choice(v.action_list)
    # 在状态s下执行动作a
    s = v.state
    s.drop_piece(a[0], a[1], 'X')
    # 创建一个新节点v'，并将v'的父节点赋值为v
    v_ = Node(s, v)
    # 将v'加入v的子节点集合
    v.children[a] = v_
    # 返回未被扩展的后继结点v'
    return v_

# UCB1


def UCB1(v: Node, C=1):
    """UCB1

    输入：节点v，超参数c
    输出：置信上限最大的动作a*
    """
    # 如果v没有子节点
    if (v.children == {}):
        return None
    # 选择置信上限最大的动作a*
    a = max(v.children.items(), key=lambda item: item[1].get_value(C))
    # 返回a*
    return a


# 游戏界面设置
WIDTH, HEIGHT = 600, 600
LINE_WIDTH = 5
BOARD_ROWS, BOARD_COLS = 8, 8
SQUARE_SIZE = WIDTH // BOARD_COLS

# 颜色定义
WHITE = (255, 255, 255)
LINE_COLOR = (23, 145, 135)
BG_COLOR = (28, 170, 156)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
AI_ITER_NUM = int(input('请输入AI的迭代次数：'))

# 初始化pygame
pygame.init()
game_over = False
board = Board()

# 设置界面
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("黑白棋")
board.draw_board()
# 主循环
while not game_over:
    pygame.display.update()
    player = 'X'
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            row, col = y // SQUARE_SIZE, x // SQUARE_SIZE
            if board.is_valid_move(row, col, player):
                board.drop_piece(row, col, player)
                player = 'O' if player == 'X' else 'X'
                board.draw_board()
                pygame.display.update()
            else:
                print('这里不能下棋！')
    # AI操作
    if player == 'O':
        # 创建一个当前board的副本用于搜索
        currentboard = Board()
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                currentboard.board[r][c] = board.board[r][c]
        # UCT搜索
        print('AI正在思考...')
        action = UCT_search(Node(currentboard), AI_ITER_NUM)
        if (action != None and action[0] in board.get_legal_actions('O')):
            board.drop_piece(action[0][0], action[0][1], player)
        player = 'X'
        board.draw_board()
        pygame.display.update()
        # 如果玩家无法行动，AI继续操作
        if (board.get_legal_actions('X') == []):
            player = 'O'

    # 检查游戏结束
    if board.terminal_test():
        game_over = True
        winner = board.calculate_winner()
        if winner == 'T':
            print('平局！')
        elif (winner == 'O'):
            print("白色（AI）赢了！")
        else:
            print("黑色（玩家）赢了！")

    pygame.display.update()
    # 控制帧率
    pygame.time.Clock().tick(30)

pygame.quit()
