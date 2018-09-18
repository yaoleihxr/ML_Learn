# /usr/bin/python
# -*- encoding:utf-8 -*-


import tensorflow as tf

def test_tensorboard():
    a = tf.constant(2)
    b = tf.constant(3)
    x = tf.add(a, b)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('E:/graph', sess.graph)
        print(sess.run(x))
        writer.close()
        # cmd中输入tensorboard - -logdir = "E:/graph"可打开tensorboard
        # 需要将cmd切换到相应目录，否则会报错寻找不到文件


if __name__ == '__main__':
    test_tensorboard()