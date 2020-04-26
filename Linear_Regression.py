# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")
tf.compat.v1.disable_eager_execution()


# %%
n_samples = 50
epochs = 200
training_rate = 0.01

test_x = np.linspace(0,35, n_samples)
test_y = 5* test_x + 5 * np.random.randn(n_samples)

plt.plot(test_x, test_y, 'o')

plt.show()


# %%
X = tf.compat.v1.placeholder(tf.float32)
Y = tf.compat.v1.placeholder(tf.float32)

W = tf.Variable(np.random.randn(), name = "weights")
B = tf.Variable(np.random.randn(), name = "bias")

pred = X*W + B

cost = tf.reduce_sum((pred - Y) ** 2) / (2 * n_samples)


# %%
optimizer = tf.compat.v1.train.GradientDescentOptimizer(training_rate).minimize(cost)


# %%
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    
    for epoch in range(epochs):
        for x, y in zip(test_x, test_y):
            sess.run(optimizer, feed_dict={X:x,Y:y})
            
        if not epoch % 20:
            c = sess.run(cost, feed_dict = {X:test_x, Y:test_y})
            w = sess.run(W)
            b = sess.run(B)
            print(f'epoch: {epoch:04d},c={c:.4f},w={w:.4f},b={b:.4f}')


# %%


