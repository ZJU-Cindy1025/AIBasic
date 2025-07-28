import pygame
import math

# 初始化pygame
pygame.init()

# 游戏界面设置
WIDTH, HEIGHT = 600, 600
LINE_WIDTH = 15
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = WIDTH // BOARD_COLS

# 颜色定义
WHITE = (255, 255, 255)
LINE_COLOR = (23, 145, 135)
BG_COLOR = (28, 170, 156)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# 设置界面
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("井字棋")

# 初始化游戏变量
board = [['' for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
player = 'X'
game_over = False

# 最大化搜索MaxValue


def MaxValue(board):
    # 判断游戏是否结束
    result = terminal_test()
    # 注意这里不是“加分”或“减分”，而是直接返回1、-1或0（结果）；同时注意平局和未结束的区别（平局算分，未结束继续）
    if result != '':
        if result == 'X':
            return -1
        elif result == 'O':
            return 1
        else:
            return 0
    # 一开始的时候设置最佳分数为负无穷
    v = -math.inf
    # 遍历所有能走的位置
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if board[r][c] == '':
                # 把当前位置设置为AI的棋子
                board[r][c] = 'O'
                # 递归搜索下一层，下一层是玩家的回合，执行最小化搜索
                score = MinValue(board)
                # 恢复当前位置为空
                board[r][c] = ''
                # 更新最佳分数
                v = max(score, v)
    return v

# 最小化搜索MinValue


def MinValue(board):
    # 判断游戏是否结束
    result = terminal_test()
    # 注意这里不是“加分”或“减分”，而是直接返回1、-1或0（结果）；同时注意平局和未结束的区别（平局算分，未结束继续）
    if result != '':
        if result == 'X':
            return -1
        elif result == 'O':
            return 1
        else:
            return 0
    # 一开始的时候设置最佳分数为正无穷
    v = math.inf
    # 遍历所有能走的位置
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if board[r][c] == '':
                # 把当前位置设置为玩家的棋子
                board[r][c] = 'X'
                # 递归搜索下一层，下一层是AI的回合，执行最大化搜索
                score = MaxValue(board)
                # 恢复当前位置为空
                board[r][c] = ''
                # 更新最佳分数
                v = min(score, v)
    return v

# 计算机AI选择最佳位置


def MinimaxDecision():
    # 一开始的时候设置最佳分数为负无穷
    v = -math.inf
    # 待定的最佳位置
    a = ()
    # 遍历每一个能走的位置
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if board[r][c] == '':
                # 将当前位置设置为AI的棋子
                board[r][c] = 'O'
                # 递归搜索下一层，下一层是玩家的回合，执行最小化搜索
                score = MinValue(board)
                # 恢复当前位置为空
                board[r][c] = ''
                # 如果分数更高，那么走这个位置
                if score > v:
                    v = score
                    a = (r, c)

    board[a[0]][a[1]] = 'O'

# 检查游戏是否结束以及胜者


def terminal_test():
    # 检查行
    for row in range(BOARD_ROWS):
        if board[row][0] == board[row][1] == board[row][2] != '':
            return board[row][0]

    # 检查列
    for col in range(BOARD_COLS):
        if board[0][col] == board[1][col] == board[2][col] != '':
            return board[0][col]

    # 检查对角线
    if board[0][0] == board[1][1] == board[2][2] != '':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != '':
        return board[0][2]

    # 检查是否平局
    if all(board[row][col] != '' for row in range(BOARD_ROWS) for col in range(BOARD_COLS)):
        return 'tie'

    # 未结束
    return ''

# 绘制棋盘


def draw_board():
    screen.fill(BG_COLOR)
    # 绘制线条
    for row in range(1, BOARD_ROWS):
        pygame.draw.line(screen, LINE_COLOR, (0, row * SQUARE_SIZE),
                         (WIDTH, row * SQUARE_SIZE), LINE_WIDTH)
    for col in range(1, BOARD_COLS):
        pygame.draw.line(screen, LINE_COLOR, (col * SQUARE_SIZE, 0),
                         (col * SQUARE_SIZE, HEIGHT), LINE_WIDTH)

    # 绘制棋子
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == 'X':
                pygame.draw.line(screen, RED, (col * SQUARE_SIZE + 20, row * SQUARE_SIZE + 20),
                                 ((col + 1) * SQUARE_SIZE - 20, (row + 1) * SQUARE_SIZE - 20), 3)
                pygame.draw.line(screen, RED, ((col + 1) * SQUARE_SIZE - 20, row * SQUARE_SIZE + 20),
                                 (col * SQUARE_SIZE + 20, (row + 1) * SQUARE_SIZE - 20), 3)
            elif board[row][col] == 'O':
                pygame.draw.circle(screen, BLUE, (int(
                    col * SQUARE_SIZE + SQUARE_SIZE / 2), int(row * SQUARE_SIZE + SQUARE_SIZE / 2)), 50, 3)


# 主循环
while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

        if event.type == pygame.MOUSEBUTTONDOWN and player == 'X':
            mouseX = event.pos[0]  # 鼠标点击的x坐标
            mouseY = event.pos[1]  # 鼠标点击的y坐标
            clicked_row = mouseY // SQUARE_SIZE
            clicked_col = mouseX // SQUARE_SIZE

            if board[clicked_row][clicked_col] == '':
                board[clicked_row][clicked_col] = 'X'
                player = 'O'
                if terminal_test() == '':
                    MinimaxDecision()
                    player = 'X'

    draw_board()

    pygame.display.update()

    # 检查游戏结束
    winner = terminal_test()
    if winner != '':
        game_over = True
        if winner == 'tie':
            print("平局！")
        else:
            print(f"{winner} 赢了！")

    # 控制帧率
    pygame.time.Clock().tick(30)

pygame.quit()
