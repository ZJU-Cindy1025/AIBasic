import sys
from PyQt5.QtWidgets import QApplication, QDialog, QLabel
from PyQt5.uic import loadUi
import requests

class MainDialog(QDialog):
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        # 使用loadui加载窗体
        self.ui=loadUi('Weather.ui', self)
        # 绑定信号与槽
        self.ui.queryBtn.clicked.connect(self.queryWeather)
        self.ui.clearBtn.clicked.connect(self.clearText)

    def queryWeather(self):
        cityName = self.ui.comboBox.currentText()
        cityCode = self.getCode(cityName)
        r = requests.get(
            "https://restapi.amap.com/v3/weather/weatherInfo?key=def944d538b9cf8ad1ee992fcf6cb7e1&city={}".format(
                cityCode)
        )
        if r.status_code == 200:
            data = r.json()['lives'][0]
            weatherMsg = '城市：{}\n天气：{}\n温度：{}\n风向：{}\n风力：{}\n湿度：{}\n发布时间：{}\n'.format(
                data['city'],
                data['weather'],
                data['temperature'],
                data['winddirection'],
                data['windpower'],
                data['humidity'],
                data['reporttime'],
            )
        else:
            weatherMsg = '天气查询失败，请稍后再试！'
        self.ui.textEdit.setText(weatherMsg)
        self.ui.show()

    def getCode(self, cityName):
        cityDict = {"北京": "110000",
                    "苏州": "320500",
                    "上海": "310000"}
        return cityDict.get(cityName, '101010100')

    def clearText(self):
        self.ui.textEdit.clear()
        self.ui.show()

if __name__ == '__main__':
    myapp = QApplication(sys.argv)
    myDlg = MainDialog()
    myDlg.show()
    sys.exit(myapp.exec_())
