import cv2
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
# 加载预训练的人脸检测模型
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 生成人脸数据集
print('训练...')
faces_dataset = []
labels = []
noface = []
files_train = os.listdir('lab4_data/train')
for file in tqdm(files_train):
    # 加载图片
    img = cv2.imread(os.path.join('lab4_data/train', file))
    # 转为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 识别人脸所在位置
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    # 这里默认仅选择第一张人脸
    if len(face) > 0:
        # 选择人脸位置
        x, y, w, h = face[0]
        face_region = gray[y:y+h, x:x+w]
        # 将人脸加入人脸数据集
        faces_dataset.append(cv2.resize(face_region, (600, 800)))
        # 设置人脸标签
        if (file.startswith('person1')):
            labels.append(1)
        elif (file.startswith('person2')):
            labels.append(2)
    else:
        noface.append(file)
print('未检测到人脸的图片：', noface)

# 使用Eigenfaces进行训练
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.train(faces_dataset, np.array(labels))

# 预测
print('预测...')
files_test = os.listdir('lab4_data/test')
result = {}
for file in tqdm(files_test):
    # 加载图片
    img = cv2.imread(os.path.join('lab4_data/test', file))
    # 转为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 识别人脸所在位置
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(face) > 0:
        # 选择人脸位置
        x, y, w, h = face[0]
        face_region = gray[y:y+h, x:x+w]
        # 进行人脸预测
        label = recognizer.predict(cv2.resize(face_region, (600, 800)))
        if (label[0] == 1):
            result[int(file.replace('.jpg', '').replace('test_', ''))] = (
                'person1', label[1])
        elif (label[0] == 2):
            result[int(file.replace('.jpg', '').replace('test_', ''))] = (
                'person2', label[1])
        else:
            result[int(file.replace('.jpg', '').replace('test_', ''))] = (
                'Unknown Face', label[1])
    else:
        result[int(file.replace('.jpg', '').replace('test_', ''))] = (
            'No face', None)

result = pd.DataFrame.from_dict(result, orient='index')
result = result.sort_index()
result.columns = ['label', 'score']
print(result)
