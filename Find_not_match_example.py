from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from model_nnlm import *
import Write_xls
import pickle
import data_build


#准备训练数据x和y
with open('./data/label_transfer_dict.pkl', 'rb') as f:
    dict = pickle.load(f)
print('ceshi', dict['肺经蕴热'])

# y_keras = np_utils.to_categorical(y,num_classes=40)
# print('keras',y_keras)
# print('标签个数',le.classes_,)
# print('标准化',le.transform(["肺经蕴热"]))
# print(y)

#将辩证分词保存成词典
dict_qingxi ={}
for k in dict.keys():
    x_train_temp = []
    for j in range(3, 8):
        if dict[k][j] !=dict[k][j]:
            x_train_temp.append('')
        else:
            x_train_temp.append(dict[k][j])
    dict_qingxi[k] = x_train_temp


str_dir = './y_str1'
with open(str_dir, 'rb') as f:
    y_str1 = pickle.load(f)
    y_str2 = pickle.load(f)
    y_str3 = pickle.load(f)
    y_str4 = pickle.load(f)
    y_str5 = pickle.load(f)
#五个角度的辨证，脏腑，八纲等
y_bianzheng = []
for i in range(len(y_str1)):
    y_bianzheng_temp = []
    y_bianzheng_temp.append(y_str1[i])
    y_bianzheng_temp.append(y_str2[i])
    y_bianzheng_temp.append(y_str3[i])
    y_bianzheng_temp.append(y_str4[i])
    y_bianzheng_temp.append(y_str5[i])
    y_bianzheng.append(y_bianzheng_temp)
# print('bianzheng',y_bianzheng)
save_Test_label_dir = './Test_Label'
with open(save_Test_label_dir, 'rb') as f:
    y_label = pickle.load(f)
text,labels = data_build.data_build_label('./data/bingli_exp_result/test')
Not_match_list = []
Not_match_text = []
Not_match_label = []
for i in range(len(y_label)):
    Not_match_list_temp = []
    Leibie = y_label[i]
    for j in range(5):
        str_temp = set(y_bianzheng[i][j])
        if(str_temp !=set(dict_qingxi[Leibie][j])):
            Not_match_list_temp.append(Leibie)
            Not_match_list_temp.append(y_bianzheng[i])
            Not_match_list_temp.append(dict_qingxi[Leibie])
            Not_match_list.append(Not_match_list_temp)
            Not_match_text.append(text[i])
            Not_match_label.append(labels[i])
            break
Write_xls.list_to_xls4(Not_match_list,Not_match_text,Not_match_label,"不匹配结果_2.xls")






