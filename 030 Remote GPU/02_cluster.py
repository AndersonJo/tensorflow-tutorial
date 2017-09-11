import tensorflow as tf

cluster = tf.train.ClusterSpec({'local': ['localhost:2222', 'localhost:2223']})

with tf.device('/job:local/task:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')

with tf.device('/job:local/task:1'):
    c = tf.matmul(a, b) + 100

with tf.Session('grpc://localhost:2222') as sess:
    result = sess.run(c)
    print(result)
