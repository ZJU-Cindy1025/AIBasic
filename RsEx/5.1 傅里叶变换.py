import skimage as ski
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

# 导入测试图片
img = ski.data.camera()
f = np.fft.fft2(img)    # 快速傅里叶变换算法得到频率分布
fshift = np.fft.fftshift(f)    # 默认结果中心点位置是左上角，将其转移到中间位置
fimg = np.log(np.abs(fshift))  # fft结果是复数，求其绝对值之后才是振幅

# 高通滤波
imgh = ski.filters.butterworth(img, 0.05, high_pass=True)
# 低通滤波
imgl = ski.filters.butterworth(img, 0.05, high_pass=False)

# 展示结果
fig = make_subplots(rows=2, cols=2, subplot_titles=[
                    '原图', '频谱图', '高通滤波', '低通滤波'])
for index, image_data in enumerate([img, fimg, imgh, imgl]):
    if index == 1:
        trace = px.imshow(image_data).data[0]
        trace.colorscale = 'gray'
    else:
        trace = px.imshow(image_data).data[0]
        trace.colorscale = 'gray'
    trace.coloraxis = None
    fig.add_trace(trace, row=index // 2 + 1, col=index % 2 + 1)
fig.update_yaxes(autorange='reversed')
fig.update_traces(showscale=False, selector={'type': 'heatmap'})
fig.show()
