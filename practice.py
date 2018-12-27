from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import preprocessing
le = preprocessing.MultiLabelBinarizer()
# onehot = LabelEncoder()
a = ['上中下']
# onehot.fit(a)
# print(onehot.classes_)
# print('zhuanhuan',onehot.transform(a))
le.fit(a)
b = '上'
print(le.transform(b))
temp = []
temp.append(b)
temp_a = np.empty(shape=(5,6))
print(temp_a)