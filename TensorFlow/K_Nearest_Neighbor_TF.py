import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (20.0, 10.0)

num_points_each_cluster = 10
mu1=[-0.4, 3]
covre1 = [[1.3, 0], [0,1]]

mu2 = [0.5, 0.75]
cover2 = [[2.2, 1.2], [1.8, 2.1]]

X1 = np.random.multivariate_normal(mu1, covre1, num_points_each_cluster)
X2 = np.random.multivariate_normal(mu2, cover2, num_points_each_cluster)


y1 = np.ones(num_points_each_cluster)
y2 = np.zeros(num_points_each_cluster)

plt.plot(X1[:,0], X1[:,1], 'ro', label='class 1')
plt.plot(X2[:,0], X2[:,1], 'bo', label='class 0')
plt.show()

x = np.vstack((X1, X2))
y = np.hstack((y1, y2))

print(x.shape, y.shape)

X_tf = tf.constant(x)
Y_tf = tf.constant(y)

# def predict(X_t, Y_t, x_t, k_t):
#     neg_one = tf.constant(-1.0, dtype=tf.float64)
#
#     # compute L-1 distance
#     distances = tf.reduce_sum(tf.abs(tf.subtract(X_t, x_t)), 1)
#     # to find the nearest points, we find the farthest points based on negative distances
#
#     neg_distances = tf.multiply(distances, neg_one)
#     # get indices
#     vals, index = tf.nn.top_k(neg_distances, k_t)
#
#     # slice hte lables
#
#     y_s = tf.gather(Y_t, index)
#     return y_s
#
# def grabLabel(predictions):
#     totalCounts = np.bincount(predictions.astype('int64'))
#     return np.argmax(totalCounts)
#
# # Generate test point
# test_point = np.array([0,0])
# test_tf = tf.constant(test_point, dtype=tf.float64)
#
# # plt.plot( X1[:, 0], X1[:,1], 'ro', label='class 1')
# # plt.plot(X2[:, 0], X2[:,1], 'bo', label='class 0')
# plt.plot(test_point[0], test_point[1], 'b', marker='D', markersize=5, label='test point')
# plt.legend(loc='best')
# plt.show()
#
# k_tf = tf.constant(3)
# pr = predict(X_tf, Y_tf, test_tf, k_tf)
# sess = tf.Session()
# y_index = sess.run(pr)
# print(grabLabel(y_index))
#
# example_2 = np.array([0.1, 2.5])
# example_2_tf = tf.constant(example_2)
# plt.plot( X1[:, 0], X1[:,1], 'ro', label='class 1')
# plt.plot(X2[:, 0], X2[:,1], 'bo', label='class 0')
# plt.plot(example_2[0], example_2[1], 'g', marker='D', markersize=10, label='test point')
# plt.legend(loc='best')
# plt.show()
#
# pr = predict(X_tf, Y_tf, test_tf, k_tf)
# y_index = sess.run(pr)
# print(grabLabel(y_index))

# https://ensemblearner.github.io/blog/2017/04/01/knn