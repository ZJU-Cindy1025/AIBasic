import cv2
import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm

# 加载预训练的人脸检测模型
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
lfw_images = ski.data.lfw_subset()
fig = make_subplots(rows=5, cols=8)
for i in tqdm(range(80, 120)):
    img = ski.util.img_as_ubyte(lfw_images[i])
    img = cv2.resize(img, (500, 500))
    # 检测图像中的人脸
    faces = face_cascade.detectMultiScale(img)
    # 在检测到的人脸周围画矩形
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), 255, 10)
    # 显示结果
    fig.add_trace(px.imshow(img).data[0], row=(
        i - 80) // 8 + 1, col=(i - 80) % 8 + 1)

fig.show()
