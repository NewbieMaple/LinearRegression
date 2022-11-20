import streamlit as st
import numpy as np
import pandas as pd
from linearRegression import linear_regression


st.write(
    """
    # 🧮 LinearRegression App
    Upload your experiment results to see the prediction of your LinearRegression .
    """
)

file_style = st.radio(
    "Select the type of file to upload or use the example file",
    ('your file', 'example file')
)
st.write("_The data requirement is that the first few columns are features and the last column is label_")


uploaded_file = st.file_uploader("Choose a file")  # 接受单个文件，

if uploaded_file is not None:  # 获取到文件的话
    st.write('prediction')
    file = np.loadtxt(uploaded_file, delimiter=',', dtype=np.float64)  # 提取文件

    # 这个要求上传文件的格式：数据要求是前面几列都是特征，保证最后一列是label
    x_data = file[:, :-1]  # 取出所有的特征,除了最后一列
    y_data = file[:, [-1]]  # 取出对应的 label

    pred, label = linear_regression(x_data, y_data)  # 用线性回归去预测

    chart = pd.DataFrame(  # 绘制图像
        np.hstack((pred, label)),  # （n，2），n是预测的个数，2代表pred和label
        columns=['predict', 'true']  # legend
    )
    st.line_chart(chart)  # 绘图


else:  # 没有获取文件的话，返回预定义的波士顿的demo

    st.write("Boston House Price Forecast Demo")
    pred, label = linear_regression()

    chart = pd.DataFrame(
        np.hstack((pred, label)),
        columns=['predict', 'true']
    )
    st.line_chart(chart)
