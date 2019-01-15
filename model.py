
import pickle

import os
import tensorflow as tf
import math
from sklearn.model_selection import train_test_split

data_pkl_path = "./data/data.pkl"
train_batch_size = 128
dev_batch_size = 256
test_batch_size = 256
learning_rate = 0.01
checkpoint_dir = 'ckpt/model.ckpt'
train = 1
epoch_num = 10000
keep_prob = 0.5
steps_per_print = 2
epochs_per_dev = 2
epochs_per_save = 10




vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
captcha_length = 4  # 验证码的长度
vocab_length = len(vocab)  # 0-9的文本长度
data_length = 10000  # 样本数量
data_path = "./data"  # 样本存放路径


def standardize(x):
    return (x - x.mean()) / x.std()


def load_data():
    """
    load data from pickle
    :return:
    """
    with open(data_pkl_path, "rb") as f:
        data_x = pickle.load(f)
        data_y = pickle.load(f)
        return standardize(data_x), data_y


def get_data(data_x, data_y):
    """
    将数据分为训练集、开发集、验证集
    :param data_x:
    :param data_y:
    :return:
    """
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.4, random_state=40)
    dev_x, test_x, dev_y, test_y = train_test_split(test_x, test_y, test_size=0.5, random_state=40)
    return train_x, train_y, dev_x, dev_y, test_x, test_y


def main():
    data_x, data_y = load_data()
    train_x, train_y, dev_x, dev_y, test_x, test_y = get_data(data_x, data_y)

    global_step = tf.Variable(-1, trainable=False, name='global_step')
    train_steps = math.ceil(train_x.shape[0] / train_batch_size)
    dev_steps = math.ceil(dev_x.shape[0] / dev_batch_size)
    test_steps = math.ceil(test_x.shape[0] / test_batch_size)

    # 构建三个Dataset对象
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(10000)
    train_dataset = train_dataset.batch(train_batch_size)

    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_x, dev_y))
    dev_dataset = dev_dataset.batch(dev_batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_dataset = test_dataset.batch(test_batch_size)

    # 初始化迭代器，并绑定在这个数据集上
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_initializer = iterator.make_initializer(train_dataset)
    dev_initializer = iterator.make_initializer(dev_dataset)
    test_initializer = iterator.make_initializer(test_dataset)

    # input layer
    with tf.variable_scope('inputs'):
        x, y_label = iterator.get_next()
    keep_prob = tf.placeholder(tf.float32, [])  # 神经网络构建graph的时候在模型中的占位
    y = tf.cast(x, tf.float32)  # 转化数据类型
    # CNN layers 三层卷积
    for _ in range(3):
        y = tf.layers.conv2d(y, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)  # 2维卷积计算
        y = tf.layers.max_pooling2d(y, pool_size=2, strides=2, padding='same')  # 二维池化方法
    # dense layers  2层全连接层
    y = tf.layers.flatten(y)
    y = tf.layers.dense(y, 1024, activation=tf.nn.relu)
    y = tf.layers.dropout(y, rate=keep_prob)
    y = tf.layers.dense(y, vocab_length)
    y_reshape = tf.reshape(y, [-1, vocab_length])
    y_label_reshape = tf.reshape(y_label, [-1, vocab_length])

    # loss
    cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_reshape, labels=y_label_reshape))
    # accuracy
    max_index_predict = tf.argmax(y_reshape, axis=-1)
    max_index_label = tf.argmax(y_label_reshape, axis=-1)
    correct_predict = tf.equal(max_index_predict, max_index_label)
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    # train
    train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

    # saver
    saver = tf.train.Saver()
    # iterator
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # global step
    gstep = 0

    # checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if train:
        for epoch in range(epoch_num):
            tf.train.global_step(sess, global_step_tensor=global_step)
            # train
            sess.run(train_initializer)
            for step in range(int(train_steps)):
                loss, acc, gstep, _ = sess.run([cross_entropy, accuracy, global_step, train_op],
                                               feed_dict={keep_prob: 0.5})
                # print log
                if step % steps_per_print == 0:
                    print('Global Step', gstep, 'Step', step, 'Train Loss', loss, 'Accuracy', acc)

            if epoch % epochs_per_dev == 0:
                # dev
                sess.run(dev_initializer)
                for step in range(int(dev_steps)):
                    if step % steps_per_print == 0:
                        print('Dev Accuracy', sess.run(accuracy, feed_dict={keep_prob: 1}), 'Step', step)

            # save model
            if epoch % epochs_per_save == 0:
                saver.save(sess, checkpoint_dir, global_step=gstep)
    else:
        # load model
        ckpt = tf.train.get_checkpoint_state('ckpt')
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Restore from', ckpt.model_checkpoint_path)
            sess.run(test_initializer)
            for step in range(int(test_steps)):
                if step % steps_per_print == 0:
                    print('Test Accuracy', sess.run(accuracy, feed_dict={keep_prob: 1}), 'Step', step)
        else:
            print('No Model Found')


if __name__ == '__main__':
    main()