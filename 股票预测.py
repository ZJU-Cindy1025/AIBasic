import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
def stockDemo(epoches,type):

    # 用pandas加载股票数据
    data=pd.read_csv('base_data/600004.csv',encoding='utf-8')
    #data['CLOSE_Y'] = data['CLOSE'].shift(-1)  # 预测下一个交易日的收盘价
    # 清洗数据（删除NaN值）
    data = data.dropna()


    # 特征和目标变量
    X = data[['PRE_CLOSE']]
    Y = data[['CLOSE']]

    # 数据标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    Y_scaled=scaler.fit_transform(Y)


    # 生成时间序列数据
    n_input = 30  # 使用过去30天的数据来预测
    generator = TimeseriesGenerator(X_scaled, Y_scaled, length=n_input, batch_size=1)

    # 构建不同的 LSTM 模型
    def model1():
        model = Sequential()
        model.add(LSTM(128, activation='relu', dropout=0.5,recurrent_dropout=0.5,
                       input_shape=(n_input, 1),return_sequences=True))
        model.add(LSTM(64, dropout=0.5,recurrent_dropout=0.5, activation='relu',
                       return_sequences=True))
        model.add(LSTM(1, dropout=0.5, recurrent_dropout=0.5, activation='relu',
                       return_sequences=False))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
        return model
    def model2():
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_input, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
        return model

    #选择模型
    if type==1:
        model=model1()
    elif type==2:
        model = model2()
    # 训练模型
    model.fit(generator,
              batch_size=100,
              epochs=epoches,
              validation_data=generator)

    # 预测
    X_test = X_scaled[-n_input:]  # 取最后 n_input 个数据点进行预测
    print(X_test)
    X_test = X_test.reshape((1, n_input, 1))

    yhat = model.predict(X_test)
    print(yhat)

    #将归一化的数据还原成真实数据
    predicted_price = scaler.inverse_transform(yhat)

    print("Predicted Price:", predicted_price[0][0])
    
stockDemo(epoches=100,type=2)