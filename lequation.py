# # CLASS WHICH TAKING IN 2 COORDINATES, 
# # PRINTS OUT THE EQUATION WHICH PASSES THROUGH THOSE POINTS
from matplotlib import pyplot as plt
import numpy as np

# import pandas as pd
# from pandas_datareader import data
# import datetime
# import time


class lequa:
    def __init__(self, c1,c2):
        self.c1 = c1
        self.c2 = c2
    def slope(self):
        x1,y1 = self.c1
        x2,y2 = self.c2
        slope = (y2-y1)/(x2-x1)
        return slope
    def equation(self):
        slope = self.slope()
        x1,y1 = self.c1

        mx = slope * x1
        b = y1-mx
        return 'y={}x+{}'.format(slope,b), b
    def check(self, b):
        x2, y2 = self.c2
        equation = self.equation()
        slope = self.slope()

        if y2-(slope*x2) == b:
            return True
        else:
            return False


x1 = 1.385960e+09
y1 = 527.767761
x2 = 1.386047e+09
y2 = 527.157166

le = lequa((x1,y1),(x2,y2))
equation, b = le.equation()

# print(equation)
# start_date = '2013-12-01'
# end_date = '2020-03-01'
# # Set the ticker
# ticker = 'GOOGL'
# # Get the data
# data = data.get_data_yahoo(ticker, start_date, end_date)

# data.reset_index(level=0, inplace=True)
# #data['Close'].plot()

# print(data.shape)
# # test_x = np.linspace(0,35, n_samples)
# # test_y = 5* test_x + 5 * np.random.randn(n_samples)

# # plt.plot(test_x, test_y, 'o')
# data['Timestamp'] = data['Date'].apply(lambda date: time.mktime(datetime.datetime.strptime(str(date),"%Y-%m-%d %H:%M:%S").timetuple()))
# test_x = data['Timestamp']
# test_y = data["Close"]

# x1 = test_x[0]
# y1 = test_y[0]
# x2 = test_x.iloc[-1]
# y2 = test_y.iloc[-1]
# x3 = test_x[785]
# y3 = test_y[785]

# le1 = lequa((x1,y1),(x2,y2))
# equation1, b1 = le1.equation()
# slope1 = le1.slope()

# le2 = lequa((x1,y1),(x3,y3))
# equation1, b2 = le2.equation()
# slope2 = le2.slope()

# le3 = lequa((x2,y2),(x3,y3))
# equation3, b3 = le3.equation()
# slope3 = le3.slope()

# slope = (slope1+slope2+slope3)/3
# b = (b1+b2+b3)/3

# equation = 'y={}x + {}'.format(slope,b)
# print(equation)