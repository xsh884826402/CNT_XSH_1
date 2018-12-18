from run_nnlm import *
import tensorflow as tf
import pickle
from data_build import *
save_dir1 = './checkpoints/zangfu/'
save_dir2 = './checkpoints/QiXueJinYe/'
save_dir3 = './checkpoints/BaGang/'
save_dir4 = './checkpoints/WeiQiYingXue/'
save_dir5 = './checkpoints/SanJiao/'
#脏腑 13个类别 气血津液 11个类别 八纲10个类别 卫气营血 4个 三焦 3个
#在第一个Test中 保存y_label 到‘./testLaebl“
# def predict():
#     # print('Hello')
#训练
# load_fenci.py 加载
cidian_dir = './cidian_data15_size7.pkl'
config = Config(15,13)#num_epoch, num_classes
model1 = TextCnn(config)
train1(model1, save_dir1,cidian_dir)
y_str1 =test1(model1, save_dir1,cidian_dir)


# print(y_return, np.shape(y_return))
_, y_test = data_build_label('./data/bingli_exp_result/test')
save_Test_label_dir = './Test_Label'
with open(save_Test_label_dir, 'wb') as f:
    pickle.dump(y_test, f)
    # print('Store Label Success')
print('Before Test1')

tf.reset_default_graph()

config = Config(15,11)
model2 = TextCnn(config)
train2(model2, save_dir2,cidian_dir)
y_str2 = test2(model2, save_dir2,cidian_dir)

tf.reset_default_graph()

config = Config(20,10)
model3 = TextCnn(config)
train3(model3, save_dir3,cidian_dir)
y_str3 = test3(model3, save_dir3,cidian_dir)

tf.reset_default_graph()
config = Config(20, 4)
model4 = TextCnn(config)
train4(model4, save_dir4,cidian_dir)
y_str4 = test4(model4, save_dir4,cidian_dir)

tf.reset_default_graph()
config = Config(20, 3)
model5 = TextCnn(config)
train5(model5, save_dir5,cidian_dir)
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

