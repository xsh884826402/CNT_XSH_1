#程序说明
#get_multilabel 和get_inverse使用的是sklearn将标签向量化
#get_multilabel_by_multiclass 分别存储五个辨证的转化词典
#get_Multiclass 将八纲里面的每一种出现的组合 处理成一个类别
#
#后来依据苗总要求，将五个中间辨证中的组合提取出来，转化为多类别标签
#以脏腑为例，解释get_MultiLabel_ZangFu_Multiclass
#get_multilabel_ZangFu_Multiclass
#   输入：待转换的y
#   功能：通过读取label——transfer_dict（字典，key是四十个分类里的类别，value是一个列表，包含五个中间辨证的具体值）,得到脏腑的一个列表。
#           用LabelBinarier去得到向量化的结果，这里LabelBinarier会自动去重。存储脏腑的10几个类别的列表到ZangFU_multiclass.pkl，这个在test时候会用到
#   输出：将对应的y转化成向量
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
import pickle
import numpy as np
from data_build import *

def get_multilabel_ZangFu(y_train):
    labels = ['心肝脾肺肾胆胃']
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    with open('./data/label_transfer_dict.pkl', 'rb') as f:
        label_transfer_dict = pickle.load(f)

    y_temp = []
    count = 0
    for item in y_train:
            # print(item)
            temp = label_transfer_dict[item][3]
            if temp != temp:
                y_temp.append('')
            else:
                y_temp.append(label_transfer_dict[item][3])
    print(y_temp)
    print(mlb.classes_)
    y_multilabel = mlb.transform(y_temp)
    print(y_multilabel)
    print('type',type(y_multilabel))
    print('shape', np.shape(y_multilabel))
    print(mlb.inverse_transform(y_multilabel))
    return y_multilabel
def get_inverse_multilabel_ZangFu(y):
    labels = ['心肝脾肺肾胆胃']
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)

    y_label = mlb.inverse_transform(y)
    return y_label
def get_multilabel_BaGang(y_train):
    labels = ['阴阳表里虚实寒热']
    index = 5
    #index 表明在哪个角度去看辩证
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    with open('./data/label_transfer_dict.pkl', 'rb') as f:
        label_transfer_dict = pickle.load(f)

    y_temp = []
    count = 0
    for item in y_train:
            # print(item)
            temp = label_transfer_dict[item][index]
            if temp != temp:
                y_temp.append('')
            else:
                y_temp.append(label_transfer_dict[item][index])
    print(y_temp)
    print('in_get_multilabel',mlb.classes_)
    y_multilabel = mlb.transform(y_temp)
    print(y_multilabel)
    return y_multilabel
def get_inverse_multilabel_BaGang(y):
    labels = ['阴阳表里虚实寒热']
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)

    y_label = mlb.inverse_transform(y)
    return y_label
def get_multilabel_QiXueJinYe(y_train):
    labels = ['气','血','湿','痰','泛','水','瘀']
    index = 4
    #index 表明在哪个角度去看辩证
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    with open('./data/label_transfer_dict.pkl', 'rb') as f:
        label_transfer_dict = pickle.load(f)

    y_temp = []
    count = 0
    for item in y_train:
            # print(item)
            temp = label_transfer_dict[item][index]
            if temp != temp:
                y_temp.append('')
            else:
                y_temp.append(label_transfer_dict[item][index])
    print(y_temp)
    print('in_get_multilabel',mlb.classes_)
    y_multilabel = mlb.transform(y_temp)
    print(y_multilabel)
    return y_multilabel
def get_inverse_multilabel_QiXueJinye(y):
    labels = ['气', '血', '湿', '痰', '泛', '水', '瘀']
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)

    y_label = mlb.inverse_transform(y)
    return y_label
def get_multilabel_WeiQiYingXue(y_train):
    labels = ['卫','气','血']
    index = 6
    #index 表明在哪个角度去看辩证
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    with open('./data/label_transfer_dict.pkl', 'rb') as f:
        label_transfer_dict = pickle.load(f)

    y_temp = []
    count = 0
    for item in y_train:
            # print(item)
            temp = label_transfer_dict[item][index]
            if temp != temp:
                y_temp.append('')
            else:
                y_temp.append(label_transfer_dict[item][index])
    print(y_temp)
    print('in_get_multilabel',mlb.classes_)
    y_multilabel = mlb.transform(y_temp)
    print(y_multilabel)
    return y_multilabel
def get_inverse_multilabel_WeiQiYingXue(y):
    labels = ['卫', '气', '血']
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)

    y_label = mlb.inverse_transform(y)
    return y_label
def get_multilabel_SanJiao(y_train):
    labels = ['上','中','下']
    index = 7
    #index 表明在哪个角度去看辩证
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    with open('./data/label_transfer_dict.pkl', 'rb') as f:
        label_transfer_dict = pickle.load(f)

    y_temp = []
    count = 0
    for item in y_train:
            # print(item)
            temp = label_transfer_dict[item][index]
            if temp != temp:
                y_temp.append('')
            else:
                y_temp.append(label_transfer_dict[item][index])
    print(y_temp)
    print('in_get_multilabel',mlb.classes_)
    y_multilabel = mlb.transform(y_temp)
    print(y_multilabel)
    return y_multilabel
def get_inverse_multilabel_SanJiao(y):
    labels = ['上', '中', '下']
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)

    y_label = mlb.inverse_transform(y)
    return y_label
def get_multilabel_by_multiclass():

    dict_transfer_ZangFu ={}
    dict_transfer_QiXueJinYe = {}
    dict_transfer_BaGang = {}
    dict_tranfer_WeiQiYingXue = {}
    dict_transfer_SanJiao = {}
    list_ZangFu = []
    list_QiXueJinYe = []
    list_BaGang = []
    list_WeiQiYingXue = []
    list_SanJiao = []

    with open('./data/label_transfer_dict.pkl', 'rb') as f:
        dict_transfer = pickle.load(f)
    for key in dict_transfer.keys():
        ZangFu_temp =dict_transfer[key][3]
        if ZangFu_temp != ZangFu_temp:
            str_temp = ''
        else:
            str_temp = ZangFu_temp
        dict_transfer_ZangFu[key] = str_temp
        list_ZangFu.append(str_temp)

#2018年12月11日
def get_multilabel_ZangFu_Multiclass(y_train):
    mlb = LabelBinarizer()
    list_to_fit = []
    with open('./data/label_transfer_dict.pkl', 'rb') as f:
        label_transfer_dict = pickle.load(f)
    for key in label_transfer_dict:
        temp = label_transfer_dict[key][3]
        if temp!= temp:
            list_to_fit.append('')
        else:
            list_to_fit.append(temp)
    mlb.fit(list_to_fit)
    # print('Classes',mlb.classes_)
    classes = mlb.classes_
    # print('class',classes,classes[1],classes.tolist())
    # print(mlb.transform(classes))
    ZangFu_multiclass_dir ='./data/ZangFu_multiclass.pkl'
    with open(ZangFu_multiclass_dir, 'wb') as f:
        pickle.dump(classes, f)

    y_temp = []
    count = 0
    for item in y_train:
            temp = label_transfer_dict[item][3]
            if temp != temp:
                y_temp.append('')
            else:
                y_temp.append(temp)
    y_multilabel = mlb.transform(y_temp)
    return y_multilabel
def get_multilabel_QiXueJinYe_Multiclass(y_train):
    mlb = LabelBinarizer()
    list_to_fit = []
    with open('./data/label_transfer_dict.pkl', 'rb') as f:
        label_transfer_dict = pickle.load(f)
    for key in label_transfer_dict:
        temp = label_transfer_dict[key][4]
        if temp!= temp:
            list_to_fit.append('')
        else:
            list_to_fit.append(temp)
    mlb.fit(list_to_fit)
    # print('Classes',mlb.classes_)
    classes = mlb.classes_
    print('Class',classes,len(classes))
    # print('class',classes,classes[1],classes.tolist())
    # print(mlb.transform(classes))
    ZangFu_multiclass_dir ='./data/QiXueJinYe_multiclass.pkl'
    with open(ZangFu_multiclass_dir, 'wb') as f:
        pickle.dump(classes, f)

    y_temp = []
    count = 0
    for item in y_train:
            temp = label_transfer_dict[item][4]
            if temp != temp:
                y_temp.append('')
            else:
                y_temp.append(temp)
    y_multilabel = mlb.transform(y_temp)
    return y_multilabel
def get_multilabel_BaGang_Multiclass(y_train):
    mlb = LabelBinarizer()
    list_to_fit = []
    with open('./data/label_transfer_dict.pkl', 'rb') as f:
        label_transfer_dict = pickle.load(f)
    for key in label_transfer_dict:
        temp = label_transfer_dict[key][5]
        if temp!= temp:
            list_to_fit.append('')
        else:
            list_to_fit.append(temp)
    mlb.fit(list_to_fit)
    # print('Classes',mlb.classes_)
    classes = mlb.classes_
    print('Class',classes,len(classes))
    # print('class',classes,classes[1],classes.tolist())
    # print(mlb.transform(classes))
    ZangFu_multiclass_dir ='./data/BaGang_multiclass.pkl'
    with open(ZangFu_multiclass_dir, 'wb') as f:
        pickle.dump(classes, f)

    y_temp = []
    count = 0
    for item in y_train:
            temp = label_transfer_dict[item][5]
            if temp != temp:
                y_temp.append('')
            else:
                y_temp.append(temp)
    y_multilabel = mlb.transform(y_temp)
    return y_multilabel
def get_multilabel_WeiQiYingXue_Multiclass(y_train):
    mlb = LabelBinarizer()
    list_to_fit = []
    with open('./data/label_transfer_dict.pkl', 'rb') as f:
        label_transfer_dict = pickle.load(f)
    for key in label_transfer_dict:
        temp = label_transfer_dict[key][6]
        if temp!= temp:
            list_to_fit.append('')
        else:
            list_to_fit.append(temp)
    mlb.fit(list_to_fit)
    # print('Classes',mlb.classes_)
    classes = mlb.classes_
    print('Class',classes,len(classes))
    # print('class',classes,classes[1],classes.tolist())
    # print(mlb.transform(classes))
    ZangFu_multiclass_dir ='./data/WeiQiYingXue_multiclass.pkl'
    with open(ZangFu_multiclass_dir, 'wb') as f:
        pickle.dump(classes, f)

    y_temp = []
    count = 0
    for item in y_train:
            temp = label_transfer_dict[item][6]
            if temp != temp:
                y_temp.append('')
            else:
                y_temp.append(temp)
    y_multilabel = mlb.transform(y_temp)
    return y_multilabel
def get_multilabel_SanJiao_Multiclass(y_train):
    mlb = LabelBinarizer()
    list_to_fit = []
    with open('./data/label_transfer_dict.pkl', 'rb') as f:
        label_transfer_dict = pickle.load(f)
    for key in label_transfer_dict:
        temp = label_transfer_dict[key][7]
        if temp!= temp:
            list_to_fit.append('')
        else:
            list_to_fit.append(temp)
    mlb.fit(list_to_fit)
    # print('Classes',mlb.classes_)
    classes = mlb.classes_
    print('Class',classes,len(classes))
    # print('class',classes,classes[1],classes.tolist())
    # print(mlb.transform(classes))
    ZangFu_multiclass_dir ='./data/SanJiao_multiclass.pkl'
    with open(ZangFu_multiclass_dir, 'wb') as f:
        pickle.dump(classes, f)

    y_temp = []
    count = 0
    for item in y_train:
            temp = label_transfer_dict[item][7]
            if temp != temp:
                y_temp.append('')
            else:
                y_temp.append(temp)
    y_multilabel = mlb.transform(y_temp)
    return y_multilabel



y = np.array([0, 0.,0.,1,0.,0,0])
print(np.reshape(y,(1,7)))
# get_inverse_multilabel_ZangFu(y)
if __name__ =="__main__":
    get_multilabel_BaGang_Multiclass([''])