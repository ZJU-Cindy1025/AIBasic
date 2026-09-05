import numpy as np
import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
# 读取示例图片
image = ski.data.camera()

# K-T变换


def kt_transform(image, k, t):
    rows, cols = image.shape
    # 生成K-T矩阵
    kt_matrix = np.zeros((rows, cols), dtype=complex)
    for i in range(rows):
        for j in range(cols):
            kt_matrix[i, j] = np.exp(1j * (k * i + t * j))
    # K-T变换
    kt_image = np.fft.fft2(image) * kt_matrix
    kt_image = np.fft.ifft2(kt_image)
    kt_image = np.abs(kt_image)
    return kt_image


# 显示结果
fig = make_subplots(rows=1, cols=3, subplot_titles=[
                    '原图', 'K=0.1, T=0.1', 'K=0.1, T=0.5'])
for column, image_data in enumerate([image, kt_transform(image, 0.1, 0.1), kt_transform(image, 0.1, 0.5)], 1):
    fig.add_trace(px.imshow(
        image_data).data[0], row=1, col=column)
fig.update_yaxes(autorange='reversed')
fig.update_traces(colorscale='gray', coloraxis=None,
                  showscale=False, selector={'type': 'heatmap'})
fig.show()
