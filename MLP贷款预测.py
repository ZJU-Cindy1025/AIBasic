from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


def MLP():
    # 输入数据（特征）
    x = [[16., 0., 0.],
         [24., 5000., 0],
         [23., 6000., 10],
         [28., 5000., 50],
         [40., 15000., 200]
         ]

    # 标签数据（标签）
    y = [0, 1, 1, 0, 1]  # 0=不放贷  1=放贷

    scaler = MinMaxScaler()  # 归一化方法，默认0~1
    scaler.fit(x)

    # 创建分类器
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(6,),
                        random_state=1, max_iter=100)

    # 训练分类器
    clf.fit(x, y)

    '''
    #更多可供设置参数
    MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=False, epsilon=1e-08,
                  hidden_layer_sizes=(5, 2), learning_rate='constant',
                  learning_rate_init=0.001, max_iter=200,
                  momentum=0.9, nesterovs_momentum=True,
                  power_t=0.5, random_state=1, shuffle=True, solver='adam',
                  tol=0.0001, validation_fraction=0.1, verbose=False,
                  warm_start=False)'''

    # 预测未知数据
    results = clf.predict([[50., 12000., 30]])
    print(results)


MLP()
