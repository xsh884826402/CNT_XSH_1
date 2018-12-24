<<<<<<< HEAD

import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from model_nnlm import *
from keras.models import Sequential
import keras.layers as ly
from keras.utils import np_utils
from sklearn import svm
import Write_xls
import pickle


#准备训练数据x和y
with open('./data/label_transfer_dict.pkl', 'rb') as f:
    dict = pickle.load(f)
a = list(dict.keys())
print(len(a),type(a))
le = preprocessing.LabelEncoder()
le.fit(a)

y = le.transform(a)
# print(y,np.shape(y))

# y_keras = np_utils.to_categorical(y,num_classes=40)
# print('keras',y_keras)
# print('标签个数',le.classes_,)
# print('标准化',le.transform(["肺经蕴热"]))
# print(y)
clf = MLPClassifier()
x = []
mlb1 = preprocessing.MultiLabelBinarizer()
mlb2 = preprocessing.MultiLabelBinarizer()
mlb3 = preprocessing.MultiLabelBinarizer()
mlb4 = preprocessing.MultiLabelBinarizer()
mlb5 = preprocessing.MultiLabelBinarizer()
mlb1.fit(['心肝脾肺肾胆胃'])
mlb2.fit(['气','血','湿','痰','泛','水','瘀'])
mlb3.fit(['阴阳表里虚实寒热'])
mlb4.fit(['卫','气','血'])
mlb5.fit(['上','中','下'])
def x_to_vector(x):
    #x 是一个一维的列表 列表中分为五个元素
    label1 = x[0]
    if label1 != label1:
        label1 = ''
    x_temp = mlb1.transform([label1])
    x_temp = np.reshape(x_temp, x_temp.size)
    x1 = x_temp

    label2 = x[1]
    if label2 != label2:
        label2 = ''
    x_temp = mlb2.transform([label2])
    x_temp = np.reshape(x_temp, x_temp.size)
    x1 = np.append(x1, x_temp)

    label3 = x[2]
    if label3 != label3:
        label3 = ''
    x_temp = mlb3.transform([label3])
    x_temp = np.reshape(x_temp, x_temp.size)
    x1 = np.append(x1, x_temp)

    label4 = x[3]
    if label4 != label4:
        label4 = ''
    x_temp = mlb4.transform([label4])
    x_temp = np.reshape(x_temp, x_temp.size)
    x1 = np.append(x1, x_temp)

    label5 = x[4]
    if label5 != label5:
        label5 = ''
    x_temp = mlb5.transform([label5])
    x_temp = np.reshape(x_temp, x_temp.size)
    x1 = np.append(x1, x_temp)

    return x1
x_train = []

for k in dict.keys():
    x_train_temp = []
    for j in range(3, 8):
        if dict[k][j] !=dict[k][j]:
            x_train_temp.append('')
        else:
            x_train_temp.append(dict[k][j])
    x_train.append(x_to_vector(x_train_temp))

x_train = np.array(x_train)
# print(x_train)

#构建模型
def train(x, y):
    model = Sequential()
    print(np.shape(x))
    model.add(ly.Dense(64,input_dim= 25,activation='sigmoid'))
    model.add(ly.Dense(64, activation='sigmoid'))
    model.add(ly.Dense(64, activation='sigmoid'))
    model.add(ly.Dense(64, activation='sigmoid'))
    model.add(ly.Dense(40, activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer='sgd',metrics=['accuracy'])
    model.fit(x, y, epochs=100)
    print('why')



    return model
def test(x, model):
    print(x)
    y = model.predict(x)
    print(y)
def train_svc(x, y):
    model = svm.SVC()
    model.fit(x, y)
    return model
'''
# model = train(x_train, y_keras)
# x_test = ['', '气', '虚', '', '中']
# x_test = x_to_vector(x_test)
# print('X-test',x_test,np.shape(x_test))
# x_test = np.reshape(x_test, (1,25))
# y_result = model.predict(x_test)
# print(y_result)
# label = np.argmax(y_result, 1)
# print(label)
# print(le.inverse_transform(label))
# x_test = ['肺', '','热', '', '']
# x_test1 = x_to_vector(x_test)
# test([list(x_test1)],model)
'''
model = train_svc(x_train, y)
x_test = ['胃', '泛', '虚', '','']
x_test = x_to_vector(x_test)
print('X-test',x_test,np.shape(x_test))
x_test = np.reshape(x_test, (1,28))
y_result = model.predict(x_test)
print(model.score(x_train, y))
print('result', y_result)
print(le.inverse_transform(y_result))


str_dir = './y_str1'
with open(str_dir, 'rb') as f:
    y_str1 = pickle.load(f)
    y_str2 = pickle.load(f)
    y_str3 = pickle.load(f)
    y_str4 = pickle.load(f)
    y_str5 = pickle.load(f)
# print(y_str1[0], y_str2[0], y_str3[0], y_str4[0])
# print(y_str1[18], y_str2[18], y_str3[18], y_str4[18])


y_trans1 = mlb1.transform(y_str1)
y_trans2 = mlb2.transform(y_str2)
y_trans3 = mlb3.transform(y_str3)
y_trans4 = mlb4.transform(y_str4)
y_trans5 = mlb5.transform(y_str5)




input_x = []
for i in range(len(y_trans1)):
    x_temp = []
    x_temp.extend(list(y_trans1[i]))
    x_temp.extend(list(y_trans2[i]))
    x_temp.extend(list(y_trans3[i]))
    x_temp.extend(list(y_trans4[i]))
    x_temp.extend(list(y_trans5[i]))
    input_x.append(x_temp)
# print(input_x, type(input_x), np.shape(input_x))
print('input_x', input_x[0])
input_y = model.predict(input_x)
y_pred = le.inverse_transform(input_y)
print(np.shape(input_y), y_pred)
save_Test_label_dir = './Test_Label'
with open(save_Test_label_dir, 'rb') as f:
    y_label = pickle.load(f)
print(y_label)
print(accuracy_score(y_label, y_pred), len(y_label))
y_different = []
y_label_test = np.asarray(y_label)

for i in range(len(y_label)):
    if y_label[i] != y_pred[i]:
        y_temp = []
        y_temp.append(y_label[i])
        y_temp.append(y_pred[i])
        y_different.append(y_temp)
# print(y_different)
=======

import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from model_nnlm import *
from keras.models import Sequential
import keras.layers as ly
from keras.utils import np_utils
from sklearn import svm
import Write_xls
import pickle


#准备训练数据x和y
with open('./data/label_transfer_dict.pkl', 'rb') as f:
    dict = pickle.load(f)
a = list(dict.keys())
print(len(a),type(a))
le = preprocessing.LabelEncoder()
le.fit(a)

y = le.transform(a)
# print(y,np.shape(y))

# y_keras = np_utils.to_categorical(y,num_classes=40)
# print('keras',y_keras)
# print('标签个数',le.classes_,)
# print('标准化',le.transform(["肺经蕴热"]))
# print(y)
clf = MLPClassifier()
x = []
mlb1 = preprocessing.MultiLabelBinarizer()
mlb2 = preprocessing.MultiLabelBinarizer()
mlb3 = preprocessing.MultiLabelBinarizer()
mlb4 = preprocessing.MultiLabelBinarizer()
mlb5 = preprocessing.MultiLabelBinarizer()
mlb1.fit(['心肝脾肺肾胆胃'])
mlb2.fit(['气','血','湿','痰','泛','水','瘀'])
mlb3.fit(['阴阳表里虚实寒热'])
mlb4.fit(['卫','气','血'])
mlb5.fit(['上','中','下'])
def x_to_vector(x):
    #x 是一个一维的列表 列表中分为五个元素
    label1 = x[0]
    if label1 != label1:
        label1 = ''
    x_temp = mlb1.transform([label1])
    x_temp = np.reshape(x_temp, x_temp.size)
    x1 = x_temp

    label2 = x[1]
    if label2 != label2:
        label2 = ''
    x_temp = mlb2.transform([label2])
    x_temp = np.reshape(x_temp, x_temp.size)
    x1 = np.append(x1, x_temp)

    label3 = x[2]
    if label3 != label3:
        label3 = ''
    x_temp = mlb3.transform([label3])
    x_temp = np.reshape(x_temp, x_temp.size)
    x1 = np.append(x1, x_temp)

    label4 = x[3]
    if label4 != label4:
        label4 = ''
    x_temp = mlb4.transform([label4])
    x_temp = np.reshape(x_temp, x_temp.size)
    x1 = np.append(x1, x_temp)

    label5 = x[4]
    if label5 != label5:
        label5 = ''
    x_temp = mlb5.transform([label5])
    x_temp = np.reshape(x_temp, x_temp.size)
    x1 = np.append(x1, x_temp)

    return x1
x_train = []

for k in dict.keys():
    x_train_temp = []
    for j in range(3, 8):
        if dict[k][j] !=dict[k][j]:
            x_train_temp.append('')
        else:
            x_train_temp.append(dict[k][j])
    x_train.append(x_to_vector(x_train_temp))

x_train = np.array(x_train)
# print(x_train)

#构建模型
def train(x, y):
    model = Sequential()
    print(np.shape(x))
    model.add(ly.Dense(64,input_dim= 25,activation='sigmoid'))
    model.add(ly.Dense(64, activation='sigmoid'))
    model.add(ly.Dense(64, activation='sigmoid'))
    model.add(ly.Dense(64, activation='sigmoid'))
    model.add(ly.Dense(40, activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer='sgd',metrics=['accuracy'])
    model.fit(x, y, epochs=100)
    print('why')



    return model
def test(x, model):
    print(x)
    y = model.predict(x)
    print(y)
def train_svc(x, y):
    model = svm.SVC()
    model.fit(x, y)
    return model
'''
# model = train(x_train, y_keras)
# x_test = ['', '气', '虚', '', '中']
# x_test = x_to_vector(x_test)
# print('X-test',x_test,np.shape(x_test))
# x_test = np.reshape(x_test, (1,25))
# y_result = model.predict(x_test)
# print(y_result)
# label = np.argmax(y_result, 1)
# print(label)
# print(le.inverse_transform(label))
# x_test = ['肺', '','热', '', '']
# x_test1 = x_to_vector(x_test)
# test([list(x_test1)],model)
'''
model = train_svc(x_train, y)
x_test = ['胃', '泛', '虚', '','']
x_test = x_to_vector(x_test)
print('X-test',x_test,np.shape(x_test))
x_test = np.reshape(x_test, (1,28))
y_result = model.predict(x_test)
print(model.score(x_train, y))
print('result', y_result)
print(le.inverse_transform(y_result))


str_dir = './y_str1'
with open(str_dir, 'rb') as f:
    y_str1 = pickle.load(f)
    y_str2 = pickle.load(f)
    y_str3 = pickle.load(f)
    y_str4 = pickle.load(f)
    y_str5 = pickle.load(f)
# print(y_str1[0], y_str2[0], y_str3[0], y_str4[0])
# print(y_str1[18], y_str2[18], y_str3[18], y_str4[18])


y_trans1 = mlb1.transform(y_str1)
y_trans2 = mlb2.transform(y_str2)
y_trans3 = mlb3.transform(y_str3)
y_trans4 = mlb4.transform(y_str4)
y_trans5 = mlb5.transform(y_str5)




input_x = []
for i in range(len(y_trans1)):
    x_temp = []
    x_temp.extend(list(y_trans1[i]))
    x_temp.extend(list(y_trans2[i]))
    x_temp.extend(list(y_trans3[i]))
    x_temp.extend(list(y_trans4[i]))
    x_temp.extend(list(y_trans5[i]))
    input_x.append(x_temp)
# print(input_x, type(input_x), np.shape(input_x))
print('input_x', input_x[0])
input_y = model.predict(input_x)
y_pred = le.inverse_transform(input_y)
print(np.shape(input_y), y_pred)
save_Test_label_dir = './Test_Label'
with open(save_Test_label_dir, 'rb') as f:
    y_label = pickle.load(f)
print(y_label)
print(accuracy_score(y_label, y_pred), len(y_label))
y_different = []
y_label_test = np.asarray(y_label)

for i in range(len(y_label)):
    if y_label[i] != y_pred[i]:
        y_temp = []
        y_temp.append(y_label[i])
        y_temp.append(y_pred[i])
        y_different.append(y_temp)
# print(y_different)
>>>>>>> 0788edf81b2bbb8ef6e66d48738a23171b113672
Write_xls.list_to_xls(y_different,"预测结果.xls")