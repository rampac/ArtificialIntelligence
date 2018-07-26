import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as pplt

x_data = np.random.rand(100).astype(np.float32)

y_data = x_data * 3 + 2

y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.1))(y_data)

weight = tf.Variable(1.0)
bias = tf.Variable(0.2)

y = weight * x_data + bias

loss_factor = tf.reduce_mean(tf.square(y - y_data))

linear_optimiser = tf.train.GradientDescentOptimizer(0.5)

train = linear_optimiser.minimize(loss_factor)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

train_data = []
for step in range(100):
    evals = sess.run([train, weight, bias])[1:3]
    if step % 5 == 0:
        print(step, evals)
        train_data.append(evals)

converter = pplt.colors
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb >1.0: cb = 1.0
    if cg<0.0: cg = 0.0

    [a,b] = f
    f_y = np.vectorize(lambda x:a*x + b)(x_data)

    line = pplt.plot(x_data, f_y)

    pplt.setp(line, color=(cr, cg, cb))

    pplt.plot(x_data, y_data, 'ro')

    green_line = mpatches.Patch(color='red', label='Data Points')

    pplt.legend(handles=[green_line])

    pplt.show()
