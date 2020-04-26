import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

epochs = 1000

data = pd.read_csv('stocks.csv')

data_x = np.array(data['Timestamp']).reshape(-1,1)
data_y = np.array(data['Scale Close']).reshape(-1,1)
plt.plot(data_y, label = 'Data')

reglr = linear_model.LinearRegression(n_jobs=1)
x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,test_size=0.1, random_state = 4)

def fit(epochs):
    for epoch in range(epochs):
        reglr.fit(x_train, y_train)
        print('coef:',reglr.coef_)

    pre = reglr.predict(x_test)

    print('FirstElement', pre[0],'FirstElementActual',y_test[0])

    #mean sqaure error
    mse = np.mean((pre-y_test)**2)
    print(mse)
    count = reglr.score(x_test,y_test)
    return count, pre

accuracy1, pre = fit(epochs)
accuracy2, pre = fit(epochs+1000)
print('####ACCURACY1',accuracy1,'####ACCURACY2',accuracy2)
plt.plot(pre, label = "Prediction")

print(data.head(10))

plt.legend()
plt.show()