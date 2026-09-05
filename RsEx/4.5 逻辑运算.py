import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
# 读取图像
img1 = ski.data.horse()
img2 = ski.data.checkerboard()

img1 = ski.transform.resize(img1, img2.shape)
img1 = ski.img_as_ubyte(img1)
img2 = ski.img_as_ubyte(img2)

# 反运算
img3 = 255-img1
# 或运算
img4 = img1 | img2
# 与运算
img5 = img1 & img2
# 异或运算
img6 = img1 ^ img2

# 显示图像
fig = make_subplots(rows=2, cols=3, subplot_titles=[
                    '图像1', '图像2', '反运算图像', '或运算图像', '与运算图像', '异或运算图像'])
for index, image_data in enumerate([img1, img2, img3, img4, img5, img6]):
    fig.add_trace(
        px.imshow(image_data).data[0], row=index // 3 + 1, col=index % 3 + 1)
fig.update_yaxes(autorange='reversed')
fig.update_traces(colorscale='gray', coloraxis=None,
                  showscale=False, selector={'type': 'heatmap'})
fig.show()
