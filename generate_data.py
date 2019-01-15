import os
import pickle
import random

import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha

vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
captcha_length = 4  # 验证码的长度
vocab_length = len(vocab)  # 0-9的文本长度
data_length = 10000  # 样本数量
data_path = "./data"  # 样本存放路径


def gen_captcha(captcha_text):
    """
    生成验证码array
    :param captcha_text: 验证码文本
    :return: 验证码array
    """
    image = ImageCaptcha()
    captcha = image.generate(captcha_text)
    captcha_image = Image.open(captcha)
    captcha_array = np.array(captcha_image)
    captcha_image.save("./captcha.jpg")
    return captcha_array


def text2vec(text):
    """
    文本转为one_hot编码
    :param text: 验证码文本
    :return: one_hot编码
    """
    if len(text) > captcha_length:
        return False
    vector = np.zeros(captcha_length * vocab_length)  # 0填充的array
    for i, c in enumerate(text):
        index = i * vocab_length + vocab.index(c)
        vector[index] = 1
    return vector


def vec2text(vector):
    """
    one_hot编码转换为文本
    :param vector:
    :return:
    """
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    vector = np.reshape(vector, [captcha_length, -1])
    # vector.shape = (captcha_length, vocab_length)
    text = ""
    for item in vector:
        text += vocab[np.argmax(item)]
    return text


def get_random_text():
    """获取随机验证码文本"""
    text = ""
    for i in range(captcha_length):
        text += random.choice(vocab)
    return text


def generate_data():
    """
    构造数据
    :return:
    """
    print("Generateing Data...")
    data_x, data_y = [], []  # data_x 为验证码数组列表，data_y 为验证码one_hot编码列表
    for i in range(data_length):
        text = get_random_text()
        captcha_array = gen_captcha(text)
        vector = text2vec(text)
        data_x.append(captcha_array)
        data_y.append(vector)
        print("Generateing Data...{}".format(i))
    # write data to pickle
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    x = np.asarray(data_x, dtype=np.float32)  # x为验证码数组的数组
    y = np.asarray(data_y, dtype=np.float32)  # y为验证码文本的one_hot编码数组
    with open(os.path.join(data_path, "data.pkl"), "wb") as f:
        # pickle 将数据通过特殊的形式转换为只有python语言认识的字符串，并写入文件
        pickle.dump(x, f)
        pickle.dump(y, f)


if __name__ == '__main__':
    generate_data()
