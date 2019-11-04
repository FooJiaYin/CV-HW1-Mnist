# src: https://ithelp.ithome.com.tw/articles/10186385

import tensorflow as tf
import numpy as np

# 用 numpy 亂數產生 100 個點，並且
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.) 
# 等等 tensorflow 幫我們慢慢地找出 fitting 的權重值

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
        plt.plot(x_data, y_data, 'ro', label='Original data')
        plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label='Fitted line')
        plt.legend()
        plt.show()

# Learns best fit is W: [0.1], b: [0.3]
以下就是每 20 round 印出的 W 還有 b，可以發現越來越接近原本的設定值

(0, array([-0.69138807], dtype=float32), array([ 0.36239833], dtype=float32))
(20, array([-0.28371689], dtype=float32), array([ 0.53784662], dtype=float32))
(40, array([-0.1455344], dtype=float32), array([ 0.45219433], dtype=float32))
(60, array([-0.0571136], dtype=float32), array([ 0.39738676], dtype=float32))
(80, array([-0.00053454], dtype=float32), array([ 0.36231628], dtype=float32))
(100, array([ 0.03566952], dtype=float32), array([ 0.33987522], dtype=float32))
(120, array([ 0.05883594], dtype=float32), array([ 0.32551554], dtype=float32))
(140, array([ 0.07365976], dtype=float32), array([ 0.31632701], dtype=float32))
(160, array([ 0.08314531], dtype=float32), array([ 0.31044737], dtype=float32))
(180, array([ 0.08921498], dtype=float32), array([ 0.30668509], dtype=float32))
(200, array([ 0.09309884], dtype=float32), array([ 0.30427769], dtype=float32))