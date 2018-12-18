#本脚本只负责预测结果
from run_nnlm import *
import tensorflow as tf
import pickle
import sklearn.metrics as mt
from data_build import *
save_dir1 = './checkpoints/zangfu/'
save_dir2 = './checkpoints/QiXueJinYe/'
save_dir3 = './checkpoints/BaGang/'
save_dir4 = './checkpoints/WeiQiYingXue/'
save_dir5 = './checkpoints/SanJiao/'
#脏腑 13个类别 气血津液 11个类别 八纲10个类别 卫气营血 4个 三焦 3个
# load_fenci.py 加载
cidian_dir ='./cidian_data15_size7.pkl'
config = Config(15,13)#num_epoch, num_classes
model1 = TextCnn(config)
y_str1 =test1(model1, save_dir1,cidian_dir)
_, y_test = data_build_label('./data/bingli_exp_result/test')
save_Test_label_dir = './Test_Label'
with open(save_Test_label_dir, 'wb') as f:
    pickle.dump(y_test, f)
    # print('Store Label Success')
tf.reset_default_graph()

config = Config(15,11)
model2 = TextCnn(config)
# train2(model2, save_dir2)
y_str2 = test2(model2, save_dir2,cidian_dir)

tf.reset_default_graph()

config = Config(20,10)
model3 = TextCnn(config)
# train3(model3, save_dir3)
y_str3 = test3(model3, save_dir3,cidian_dir)

tf.reset_default_graph()
config = Config(20, 4)
model4 = TextCnn(config)
# train4(model4, save_dir4)
y_str4 = test4(model4, save_dir4,cidian_dir)

tf.reset_default_graph()
config = Config(20, 3)
model5 = TextCnn(config)
# train5(model5, save_dir5)
y_str5 = test5(model5, save_dir5,cidian_dir)
#开始对病历预测结果
str_dir ='./y_str1'
#按照脏腑 气血津液，八纲，卫气营血，三焦
with open(str_dir, 'wb') as f:
    pickle.dump(y_str1, f)
    pickle.dump(y_str2, f)
    pickle.dump(y_str3, f)
    pickle.dump(y_str4, f)
    pickle.dump(y_str5, f)
    print('Store Success')

#根据得到的五个y_str（五个角度的辨证）去按照表匹配得到结果

with open('./data/label_transfer_dict.pkl', 'rb') as f:
    dict = pickle.load(f)
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
#将五个辨证拼接起来，得到中间特征
y_bianzheng = []
for i in range(len(y_str1)):
    y_bianzheng_temp = []
    y_bianzheng_temp.append(y_str1[i])
    y_bianzheng_temp.append(y_str2[i])
    y_bianzheng_temp.append(y_str3[i])
    y_bianzheng_temp.append(y_str4[i])
    y_bianzheng_temp.append(y_str5[i])
    y_bianzheng.append(y_bianzheng_temp)

y_pred = []
# for key in dict_qingxi.keys():
#     print('key', key)
#     print('dict',dict_qingxi[key] )
#     # print('dict_j',dict_qingxi[key][j])
for i in range(len(y_bianzheng)):
    flag_1 = 0
    print('y _bianzheng', i ,y_bianzheng[i])
    for key in dict_qingxi.keys():
        print('key',key)
        flag_2 = 0
        for j in range(0,5):
            if(set(dict_qingxi[key][j])!= set(y_bianzheng[i][j])):
                break
            flag_2 =flag_2+1
        if(flag_2 == 5):
            y_pred.append(key)
            flag_1 = 1
            break
    if(flag_1 == 0):
        y_pred.append('unknown')
    # print('y_predy',y_pred[i])

print('pred',y_pred)
save_Test_label_dir = './Test_Label'
with open(save_Test_label_dir, 'rb') as f:
    y_label = pickle.load(f)
print('label',y_label)
print('acc', mt.accuracy_score(y_label,y_pred))
print('length', len(y_label))

