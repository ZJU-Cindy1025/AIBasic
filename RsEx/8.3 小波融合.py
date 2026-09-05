import numpy as np
import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
import pywt
# 读取两幅图像
img1 = ski.data.astronaut()
img2 = ski.data.chelsea()

# 将img1调整为img2大小
img1 = ski.transform.resize(img1, img2.shape)

# 基于小波的图像融合
coeffs1 = pywt.dwt2(img1, wavelet='haar')
coeffs2 = pywt.dwt2(img2, wavelet='haar')
cA1, (cH1, cV1, cD1) = coeffs1
cA2, (cH2, cV2, cD2) = coeffs2
# 将两幅图像的小波系数按照一定的权重相加
cA_new = (cA1 + cA2) / 2
cH_new = (cH1 + cH2) / 2
cV_new = (cV1 + cV2) / 2
cD_new = (cD1 + cD2) / 2
coeffs_new = cA_new, (cH_new, cV_new, cD_new)
img1_new = pywt.idwt2(coeffs_new, wavelet='haar')
img1_new = img1_new.astype(np.uint8)

# 显示结果
fig = make_subplots(rows=1, cols=3, subplot_titles=['图像1', '图像2', '小波融合后图像'])
for column, image_data in enumerate([img1, img2, img1_new[:, :, :3]], 1):
    fig.add_trace(px.imshow(image_data).data[0], row=1, col=column)
fig.update_yaxes(autorange='reversed')
fig.update_traces(showscale=False, selector={'type': 'heatmap'})
fig.show()
