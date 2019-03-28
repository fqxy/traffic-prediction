import matplotlib.pyplot as plt
import matplotlib

MAE = [27.22, 27.44, 27.38]
RMSE = [40.38, 39.13, 37.55]
MAPE = [13.54, 13.82, 13.34]
x = [1, 2, 3]

matplotlib.rcParams['font.family'] = 'SimHei'

plt.bar(x, MAE, color='red')
plt.bar(x, RMSE, color='yellow' )
plt.bar(x, y3)


plt.show()