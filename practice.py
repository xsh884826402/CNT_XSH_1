from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
le = preprocessing.LabelBinarizer()
# onehot = LabelEncoder()
a = ['啊','心','肝肾','肝','肾','肾','脾胃','脾']
# onehot.fit(a)
# print(onehot.classes_)
# print('zhuanhuan',onehot.transform(a))
le.fit(a)
print('calss',le.classes_)
print(le.transform(a))