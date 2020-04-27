import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import random

plt.style.use("fivethirtyeight")
tf.compat.v1.disable_eager_execution()

epochs = 200
training_rate = 0.01

data = pd.read_csv("weather_lowell.csv")
n_samples = len(data)

# data['Timestamp'] = data['DATE'].apply(lambda date: time.mktime(datetime.datetime.strptime(str(date),"%Y-%m-%d").timetuple()))
# data['Scale Weather'] = data['TMAX'].apply(lambda tmax: tmax/data["TMAX"].max())
# data.to_csv('weather_lowell.csv')
#data['Scale Weather'].plot()
data['TMAX'].plot()

test_x = data['Timestamp']
test_y = data["Scale Weather"]

X = tf.compat.v1.placeholder(tf.float32)
Y = tf.compat.v1.placeholder(tf.float32)

W = tf.Variable(0.5, name = "weights", dtype=tf.float32)
B = tf.Variable(0, name = "bias", dtype=tf.float32)

# x = np.linspace(0,1570,100)
# plt.plot(x, W*x + B)
# plt.show()

pred = tf.math.add( tf.math.multiply(X,W), B)

cost = tf.reduce_mean((pred - Y) ** 2) / (2 * n_samples)

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

        accuracy = 0
        for i in range(n_samples):
            if round(data['Learned Answer'][i]) == round(test_y[i]):
                accuracy += 1

        data['Learned Answer'].plot()
        
        plt.legend()
        plt.show()

        test_date = '2025-01-01 00:00:00'
        timestamp = time.mktime(datetime.datetime.strptime(test_date,"%Y-%m-%d %H:%M:%S").timetuple())

        answer = timestamp * weight + bias
        print('Weather in',test_date,'will most likely be',answer*data['TMAX'].max())
        print(data, '####ACURACY:',accuracy)
except KeyboardInterrupt or Exception as e:
    if Exception:
        print(e)
    if KeyboardInterrupt:
        exit()