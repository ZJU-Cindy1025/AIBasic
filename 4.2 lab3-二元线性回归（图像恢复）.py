import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_sample_image
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm
# 读取图像
base_img = load_sample_image('flower.jpg')
# 生成受损图像
damage_img = np.copy(base_img)
# 噪声遮罩仅包含{0,1}值，对原图的噪声遮罩可以每行分别用0.8/0.4/0.6的噪声比率产生，即噪声遮罩每个通道每行80%/40%/60%的像素值为0，其他为1
# 生成三个通道的噪声遮罩
mask = np.random.choice([0, 1], size=damage_img.shape[:2] + (3,), p=[0.8, 0.2])
mask[..., 1] = np.random.choice(
    [0, 1], size=damage_img.shape[:2], p=[0.6, 0.4])
mask[..., 2] = np.random.choice(
    [0, 1], size=damage_img.shape[:2], p=[0.4, 0.6])
# 将受损图像的RGB通道的部分像素值设置为0
damage_img = damage_img * mask

# 修复受损图像
repaired_img = np.copy(damage_img)
model = LinearRegression()
replace_length = 5   # 修复像素的邻域半径
# 遍历像素
for channel in tqdm(range(damage_img.shape[2]), desc='波段'):
    for row in tqdm(range(damage_img.shape[0]), desc='行'):
        for col in range(damage_img.shape[1]):
            if np.all(mask[row, col] == 0):
                train_x = []
                train_y = []
                for dx in range(-replace_length, replace_length+1):
                    for dy in range(-replace_length, replace_length+1):
                        if 0 <= row+dy < damage_img.shape[0] and 0 <= col+dx < damage_img.shape[1] and np.all(mask[row+dy, col+dx] != 0):
                            train_x.append([row+dy, col+dx])
                            train_y.append(damage_img[row+dy, col+dx][channel])
                if len(train_x) > 0:
                    model.fit(train_x, train_y)
                    if (model.predict([[row, col]])[0] < 0):
                        repaired_img[row, col][channel] = 0
                    elif (model.predict([[row, col]])[0] > 255):
                        repaired_img[row, col][channel] = 255
                    else:
                        repaired_img[row, col][channel] = model.predict([[row, col]])[
                            0]

# 计算恢复图像与原始图像的2-范数之和
error = 0
for channel in range(damage_img.shape[2]):
    error += np.linalg.norm(np.array(repaired_img)
                            [:, :, channel]-np.array(base_img)[:, :, channel], 2)
print('恢复图像与原始图像的2-范数之和：', error)

# 显示图像
fig = make_subplots(rows=1, cols=3, subplot_titles=['原图', '受损图', '修复图'])
display_images = [
    np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))
    for image in [base_img, damage_img, repaired_img]
]
for column, image in enumerate(display_images, 1):
    fig.add_trace(px.imshow(image, binary_string=True).data[0], row=1, col=column)
fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
fig.show()
