from dataloader import X_train, Y_train, X_test, Y_test, X_val, Y_val, device, flatten
from QWLSTMModel import QWLSTMModel, get_rfweight
from quantile_forest import RandomForestQuantileRegressor
import numpy as np
from bayes_opt import BayesianOptimization
import json

quantile = 0.05  # 全局分位数


def target_loss(Y_true: np.ndarray, Y_predict: np.ndarray, quantile: float = quantile):
    """计算目标损失，用于贝叶斯优化

    Args:
        Y (np.ndarray): 实际值（标签）
        Y_predict (np.ndarray): 预测值
        quantile (float): 分位数

    Returns:
        float: 损失值
    """
    yp_big_num = np.sum(Y_true < Y_predict)
    proportion = yp_big_num / len(Y_true)

    return abs(proportion - quantile)


def calculate_loss(
    # Qmodel参数
    dropout,
    hidden_size,
    n_iter,
    lr,
    batch_size,
    tol,
    tau,
    num_layers,
    # RF参数
    n_estimators,
    min_samples_split,
    min_samples_leaf,
    max_depth,
):
    hidden_size = int(hidden_size)
    num_layers = int(num_layers)
    n_estimators = int(n_estimators)
    min_samples_leaf = int(min_samples_leaf)
    min_samples_split = int(min_samples_split)
    max_depth = int(max_depth)
    batch_size = int(batch_size)
    n_iter = int(n_iter)

    rf = RandomForestQuantileRegressor(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
    )
    rf.fit(flatten(X_train).cpu(), flatten(Y_train).cpu())
    mrfw, mrfwn = get_rfweight(rf, flatten(X_train).cpu())

    qwlstm_model = QWLSTMModel(
        hs=hidden_size, quantile=quantile, dropout=dropout, num_layers=num_layers
    )
    qwlstm_model.fit(  # 使用训练集调优超参数（70%）
        X_train,
        Y_train,
        mrfw,
        tau=tau,
        d=False,
        batch_size=batch_size,
        n_iter=n_iter,
        lr=lr,
        tol=tol,
        verbose=False,
    )

    # 预测结果
    Y_pred = qwlstm_model.predict(X_test)
    # Y_pred = y_pred1.reshape(-1, 1)  # 确保 y_pred1 是列向量

    return -target_loss(Y_test, Y_pred, quantile=quantile)


def bayesian_optimization(_iter: int = 500):
    pbounds = {
        "dropout": (0.0, 0.5),
        "hidden_size": (4, 128),
        "n_iter": (500, 2000),
        "lr": (1e-4, 1e-1),
        "batch_size": (4, 256),
        "tol": (1e-6, 1e-4),
        "tau": (0.0, 1.0),
        "num_layers": (1, 4),
        "n_estimators": (50, 200),
        "min_samples_leaf": (1, 15),
        "min_samples_split": (2, 20),
        "max_depth": (5, 30),
    }

    optimizer = BayesianOptimization(f=calculate_loss, pbounds=pbounds, random_state=1)

    optimizer.maximize(init_points=30, n_iter=_iter)  # 优化完成
    best_params = optimizer.max["params"]

    with open("best_params.txt", "w") as f:
        json.dump(best_params, f)  # 将参数保存到文件中


def load_best_params():
    with open("best_params.txt", "r") as f:
        best_params = json.load(f)

    best_params_inted = {
        "hidden_size": int(best_params["hidden_size"]),
        "num_layers": int(best_params["num_layers"]),
        "n_estimators": int(best_params["n_estimators"]),
        "min_samples_split": int(best_params["min_samples_split"]),
        "min_samples_leaf": int(best_params["min_samples_leaf"]),
        "max_depth": int(best_params["max_depth"]),
        "batch_size": int(best_params["batch_size"]),
        "n_iter": int(best_params["n_iter"]),
    }

    best_params_float = {
        "dropout": best_params["dropout"],
        "lr": best_params["lr"],
        "tol": best_params["tol"],
        "tau": best_params["tau"],
    }

    return {**best_params_inted, **best_params_float}


# def train_model_1():  # 滚动向前预测训练
#     try:
#         best_params = load_best_params()
#     except FileNotFoundError:
#         print("best_params.txt not found. Please run bayesian_optimization() first.")
#         return

#     rf = RandomForestQuantileRegressor(
#         n_estimators= best_params n_estimators,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=min_samples_leaf,
#         max_depth=max_depth,
#     )
#     rf.fit(flatten(X_train), flatten(Y_train))
#     mrfw, mrfwn = get_rfweight(rf, flatten(X_train))

#     qwlstm_model = QWLSTMModel(
#         hs=hidden_size, quantile=quantile, dropout=dropout, num_layers=num_layers
#     )
#     qwlstm_model.fit(  # 使用训练集调优超参数（70%）
#         X_train,
#         Y_train,
#         mrfw,
#         tau=tau,
#         d=False,
#         batch_size=batch_size,
#         n_iter=n_iter,
#         lr=lr,
#         tol=tol,
#         verbose=False,
#     )


if __name__ == "__main__":
    bayesian_optimization(10)
