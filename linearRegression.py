from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 导入数据
boston = load_boston()  # 导入库中自带的数据集
x = boston['data']
y = boston['target']


def linear_regression(data=x, label=y):  # 接受数据和label

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3)  # 0.3 test ，0.7 train

    model = LinearRegression()  # 导入线性回归模型
    model.fit(x_train, y_train)  # 训练数据

    y_pred = model.predict(x_test)  # 网络预测

    y_pred = y_pred.reshape(-1, 1)  # 改变成（n，1）的大小，为了streamlit的绘图
    y_test = y_test.reshape(-1, 1)

    return y_pred, y_test  # 返回预测值和真实值
