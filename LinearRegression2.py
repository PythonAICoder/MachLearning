import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data
import datetime
import time
import random

plt.style.use("fivethirtyeight")
tf.compat.v1.disable_eager_execution()

###EQUATION: ############### y=-7.018333333333211e-06x+10254.897027666497 #############
##############################OR y=4.120264489871436e-06x + -5250.689416195902#########
# n_samples = 30
epochs = 200
training_rate = 0.01
n_samples = 1570

start_date = '2013-12-01'
end_date = '2020-03-01'
# Set the ticker
ticker = 'GOOGL'
# Get the data
data = data.get_data_yahoo(ticker, start_date, end_date)

data.reset_index(level=0, inplace=True)
#data.to_csv("stocks.csv")
data = data.iloc[::-1]


# x= np.linspace(0,1570,100)
# y = 4.120264489871436e-06*x + -5250.689416195902
# plt.plot(x,y)

plt.show()

print(data.shape)
# test_x = np.linspace(0,35, n_samples)
# test_y = 5* test_x + 5 * np.random.randn(n_samples)

# plt.plot(test_x, test_y, 'o')
data['Timestamp'] = data['Date'].apply(lambda date: time.mktime(datetime.datetime.strptime(str(date),"%Y-%m-%d %H:%M:%S").timetuple()))
data['Scale Close'] = data['Close'].apply(lambda close: close/data["Close"].max())
data['Scale Close'].plot()
data.to_csv('stocks.csv')
test_x = data['Timestamp']
test_y = data["Scale Close"]

X = tf.compat.v1.placeholder(tf.float32)
Y = tf.compat.v1.placeholder(tf.float32)

W = tf.Variable(1, name = "weights", dtype=tf.float32)
B = tf.Variable(0, name = "bias", dtype=tf.float32)

# x = np.linspace(0,1570,100)
# plt.plot(x, W*x + B)
# plt.show()

pred = tf.math.add( tf.math.multiply(X,W), B)

cost = tf.reduce_sum((pred - Y) ** 2) / (2 * n_samples)

print('Weight',W, 'bias', B,'Prediction',pred,'Cost Function',cost)

optimizer = tf.compat.v1.train.AdamOptimizer(training_rate).minimize(cost)

init = tf.compat.v1.global_variables_initializer()

try:
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        start = time.time()

        for epoch in range(epochs):
            inepoch = time.time()

            for index, row in data.iterrows():
                sess.run(optimizer, feed_dict={X:test_x[index],Y:test_y[index]})
                afopt = time.time()
            c = sess.run(cost, feed_dict = {X:test_x[epoch], Y:test_y[epoch]})
            w = sess.run(W)
            b = sess.run(B)
            if not epoch % 20:
                print('Took',str(int(afopt-inepoch)),'seconds to run optimizer')
                if epoch > 20:
                    print('Every ten epochs, prog takes',str(evten-inepoch),'seconds')
                print('epoch:',epoch,'c=',c,'w=', w,'b=', b)
                evten = time.time()
                
        #plt.plot(test_x, test_y, 'o')
        #plt.plot(test_x, 5 * test_x + 5, label = "Answer")
        
        weight = sess.run(W)
        bias = sess.run(B)
        print('weight, bias=', weight, bias)

        # learned_answer = []
        # for items in test_x.iteritems():
        #     items = np.float32(items)
        #     learned_answer.append(items*weight + bias)
        # learn = pd.DataFrame({'Learned Answer': learned_answer})
        data['Learned Answer'] = test_x.apply(lambda stamp: weight * stamp + bias)

        end = time.time()
        print('Took',str(int(end-start)/60),'minutes long')

        print(data["Learned Answer"])

        data['Learned Answer'].plot()
        
        plt.legend()
        plt.show()

        test_date = '2025-01-01 00:00:00'
        timestamp = time.mktime(datetime.datetime.strptime(test_date,"%Y-%m-%d %H:%M:%S").timetuple())

        answer = timestamp * weight + bias
        print('Stock in',test_date,'will most likely be',answer*data['Close'].max())
        print(data)
except KeyboardInterrupt or Exception as e:
    if Exception:
        print(e)
    if KeyboardInterrupt:
        exit()