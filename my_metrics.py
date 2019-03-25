import numpy as np

# 平均绝对误差（MAE）
def mae_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mae -- MAE 评价指标
    """

    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred)) / n
    return mae

# 均方误差（MSE）
def mse_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mse -- MSE 评价指标
    """

    n = len(y_true)
    mse = sum(np.square(y_true - y_pred)) / n
    return mse

# 平均绝对百分比误差（MAPE）
def mape(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mape -- MAPE 评价指标
    """

    _abs = np.abs((y_true - y_pred) / y_true)

    where_are_inf = np.isinf(_abs)
    _abs[where_are_inf] = 0

    # np.savetxt('new.csv', _abs, delimiter=',')
    # _abs2list = _abs.tolist()

    mape = np.average(np.average(_abs, axis=0))

    return mape*100
