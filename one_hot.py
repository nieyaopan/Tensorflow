import numpy as np
from sklearn import preprocessing
from keras.utils import to_categorical



def fun1():
    data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot', 'pan', 'nie', 'yao']
    value = np.array(data)
    print(value)
    # integer encode
    label_encoder = preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(value)
    print(integer_encoded)

    # binary encode
    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)  # sparse = False这个参数来禁用稀疏返回类型
    integer_encoded.shape = (len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)

    # invert example
    a = np.argmax(onehot_encoded[0, :])
    inverted = label_encoder.inverse_transform([a])
    print(inverted)


def fun2():
    data = [1, 3, 2, 0, 3, 2, 2, 1, 0, 1, 4, 5, 6]
    data = np.array(data)
    print(data)
    onehot_enconde = to_categorical(data)
    print(onehot_enconde)
    print(type(onehot_enconde))
    # invert encoding
    invertd = np.argmax(onehot_enconde[1])
    print(invertd)


if __name__ == '__main__':
    fun1()
