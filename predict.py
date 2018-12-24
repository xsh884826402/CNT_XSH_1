from model_nnlm import *
from run_nnlm import *
from data_build import *
import pickle
import keras as kr
import readText
def predict():
    config = Config()
    model = TextCnn(config)
    dict_dir = open('./result/dict_11344', 'rb')
    dict = pickle.load(dict_dir)
    # X_test = [['口干 咽干 口渴 烦躁 咳嗽 苔黄']]
    X_test =readText.getText('./data/TestHe.txt')
    print('pre', X_test)
    result = []
    for str in X_test:
        result.append(str.split())
    X_test =result
    print('post', result)
    # X_test,Y_test = data_build('./data/bingli_exp_result/test')
    # X_test = [['小便不利', '小便涩痛', '尿急', '腰痛', '便溏', '阴痒', '舌红', '舌苔黄', '舌苔腻'],
    #           ['小便不利', '小便涩痛', '尿急', '腰痛', '便溏', '阴痒', '舌红', '舌苔黄', '舌苔腻'],
    #           ['小便不利', '小便涩痛', '尿急', '腰痛', '便溏', '阴痒', '舌红', '舌苔黄', '舌苔腻'],
    #           ['感冒', '恶寒', '流涕', '咽痒','咽痒', '鼻塞', '口干', '有痰', '咯不出', '舌红', '苔略黄']]
    # print('xtest', X_test)
    file_cidian = open('./cidian_data15.pkl', 'rb', )
    indexword = pickle.load(file_cidian)
    X_test = word_to_index(X_test, indexword)
    X_test = kr.preprocessing.sequence.pad_sequences(X_test, config.seq_length)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        save_path ='./checkpoints/textcnn/'
        saver.restore(sess=session, save_path=save_path)
        print('X_',X_test)
        feed_dict = {
            model.input_x: X_test,
            model.keep_prob: 1.0
        }
        y_pred_cls = session.run(model.y_pred_cls, feed_dict=feed_dict)
        print(y_pred_cls)
        for i in y_pred_cls:
            print(dict[i])
if __name__ =='__main__':
    predict()
    "str".split()