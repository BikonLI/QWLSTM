# 将数据进一步处理
from data import X, Y   # 导入处理好的X与Y
import numpy as np
from sklearn.model_selection import train_test_split


def sliding_window(X: np.ndarray, Y: np.ndarray, step: int = 30):
    if len(X) != len(Y):
        raise IndexError
    x, y = [], []
    for i in range(len(X) - step + 1):
        # print(X[i:i + step, ])
        x.append(X[i:i + step, ])
        y.append(Y[i:i + step, ])

    return np.array(x), np.array(y)


# 使用滑动窗口，获得新的X，Y
X, Y = sliding_window(X, Y, 30)

# 两次拆分，获得 训练集、测试集、验证集 7:2:1 
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=1/3, random_state=42)
