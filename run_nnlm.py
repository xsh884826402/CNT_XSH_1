#encoding=utf-8
from model_nnlm import *
import os
import pickle
from data_build import *
from keras.preprocessing import sequence
from keras.utils import np_utils
from cnews_loader import *
from practice_for_multilabel import *
from sklearn.metrics import f1_score
#from tensorflow.python.training import moving_averages
def evaluate(model,sess, x, y):
    #compute the loss and acc
    data_len = len(x)
    #batch_eval = batch_iter(x, y, 128)
    total_loss =0.0
    total_acc =0.0
    feed_dict = {
        model.input_x:x,
        model.input_y:y,
        model.keep_prob:1.0
        }
    loss, acc =sess.run([model.loss, model.acc],feed_dict =feed_dict)
    return loss, acc
    # for x_batch, y_batch in batch_eval:
    #     batch_len = len(x_batch)
    #     feed_dict = {
    #         model.input_x:x_batch,
    #         model.input_y:y_batch,
    #         model.keep_prob:1.0
    #     }
    #     loss, acc =sess.run([model.loss, model.acc],feed_dict =feed_dict)
    #     total_loss += loss*batch_len
    #     total_acc += acc * batch_len
    # return total_loss/data_len,total_acc/data_len

def word_to_index(texts, indexword):
    texts_id = []
    for text in texts:
        texts_id_1 = []
        for word in text:
            try:
                texts_id_1.append(indexword[word])
            except:
                texts_id_1.append(0)
        texts_id.append(texts_id_1)
    return texts_id
def create_multi_label(Y,label_dict):
    y = []
    return y
def accuarcy_xsh(y_label,y_test, class_num):

    x, y = np.shape(y_test)
    result = y_label==y_test

    result = np.sum(result, axis=1)
    print('result', result)
    result = result==class_num
    result_counter = np.sum(result)
    print('x',x,'result_counter',result_counter)
    return result_counter/x



def train1(model, save_dir,cidian_dir):
    #脏腑
    #save_dir 保存模型的路径
    print("config Tensorboard and saver")

    tensorboard_dir = 'C:/logs'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy",model.acc)

    #merge all
    merged_summary = tf.summary.merge_all()


    #config saver
    saver = tf.train.Saver()
    # save_dir = './checkpoints/zangfu/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("loading data")
    #input data
    #the name of the dictionary
    file_cidian = open(cidian_dir, 'rb', )
    indexword = pickle.load(file_cidian)
    vectorword = pickle.load(file_cidian)

    #the directory of the input data
    x_train, y_train = data_build_label('./data/bingli_exp_result/train')
    x_test, y_test = data_build_label('./data/bingli_exp_result/test')

    #converse word to index
    # print('x_test', x_test[1])
    x_train = word_to_index(x_train, indexword)
    x_test = word_to_index(x_test, indexword)
    # print('x_train', x_train)

    #padding the sequence
    x_train = sequence.pad_sequences(x_train, model.config.seq_length)
    x_test = sequence.pad_sequences(x_test, model.config.seq_length)

    # y_train = np_utils.to_categorical(y_train,)
    # y_test = np_utils.to_categorical(y_test,)
    #get 多标签分类，这里是不同的train函数的主要区别
    y_train = get_multilabel_ZangFu_Multiclass(y_train)
    y_test = get_multilabel_ZangFu_Multiclass(y_test)
    print("in Train1 y_train", y_train)
    #create session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    #writer.add_graph(session.graph)
    writer = tf.summary.FileWriter(tensorboard_dir, session.graph)
    print("Training and evaluate")
    total_batch = 0
    best_acc_val = 0.0
    last_improved =0
    require_improvement = 1000
    flag = False

    # feed_dict_eval = {
    #     model.input_x: x_train,
    #     model.input_y: y_train,
    #     model.keep_prob: model.config.dropout_keep_prob
    # }


    #print('x', model.input_x[0, :].eval(feed_dict=feed_dict_eval))
    for epoch in range(model.config.num_epochs):
        print('Epoch',epoch)
        batch_train = batch_iter(x_train, y_train, model.config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.keep_prob: model.config.dropout_keep_prob
            }
            if total_batch % model.config.save_per_batch:
                s = session.run(merged_summary, feed_dict =feed_dict)
                writer.add_summary(s, total_batch)
                # saver.save(sess=session, save_path=save_dir)

            if total_batch % model.config.print_per_batch==0:
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc],feed_dict= feed_dict)
                #?
                loss_val, acc_val = evaluate(model, session, x_test, y_test)
                # b=tf.summary.histogram('Loss',loss_val)
                # a=session.run(b)
                # writer.add_summary(b, total_batch)


                best_acc_val = acc_val
                last_improved = total_batch
                saver.save(sess=session, save_path=save_dir)
                improved_str = '*'
                # else:
                #     improved_str = ''

                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, '
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, improved_str))
                print()
            # list_gradient = session.run([model.compute[0]], feed_dict=feed_dict)#元组列表
            # temp = session.run([model.embedding])
            # print(temp, 'shape', )
            # values, index =list_gradient[0]#每个元组有gradient，var
            # print(list_gradient[0][0][0], np.shape(list_gradient[0][0][0]))
            # print(list_gradient[0][0][1])
            # print('0','values', np.shape(values), 'index', np.shape(index))
            session.run(model.optim, feed_dict = feed_dict)
            total_batch += 1
    print('----------------Predicting---------------')
    acc = session.run(model.acc, feed_dict={
        model.input_x: x_test,
        model.input_y: y_test,
        model.keep_prob: 1.0
    })
    print('acc',acc)
    # y_label = session.run(model.y_pred_cls, feed_dict={
    #     model.input_x : x_test,
    #     model.keep_prob : 1.0
    # })
    # print('label', y_label)
    # print(type(y_label), np.shape(y_label))
    # y_str = y_label.astype(int)
    # print(y_str)
    # y_str = get_inverse_multilabel_ZangFu(y_str)
    # print(y_str)
    # accuarcy = accuarcy_xsh(y_label,y_test,model.config.num_classes)
    # print('accuarcy', accuarcy)
    # score1 = f1_score(y_label,y_test,average='weighted')
    # score2 = f1_score(y_label, y_test,average='samples')
    # score3 = f1_score(y_label, y_test, average='micro')
    # score4 = f1_score(y_label, y_test,average='macro')
    # print('Score_weighted', score1)
    # print('Score_samples', score2)
    # print('Score_micro',score3)
    # print('Score_macro',score4)
    session.close()
def train2(model, save_dir,cidian_dir):
    #气血津液
    print("config Tensorboard and saver QiXueJinYe Traing")

    tensorboard_dir = 'C:/logs'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy",model.acc)

    #merge all
    merged_summary = tf.summary.merge_all()


    #config saver
    saver = tf.train.Saver()
    # save_dir = './checkpoints/BaGang/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("loading data")
    #input data
    #the name of the dictionary
    file_cidian = open(cidian_dir, 'rb', )
    indexword = pickle.load(file_cidian)
    vectorword = pickle.load(file_cidian)

    #the directory of the input data
    x_train, y_train = data_build_label('./data/bingli_exp_result/train')
    x_test, y_test = data_build_label('./data/bingli_exp_result/test')
    #convert y to multi_label

    # print('Y-train', y_train,type(y_train))
    # print('Y_test',y_test)
    y_temp = []
    count = 0

    #converse word to index
    x_train = word_to_index(x_train, indexword)
    x_test = word_to_index(x_test, indexword)
    # print('x_train', x_train)

    #padding the sequence
    x_train = sequence.pad_sequences(x_train, model.config.seq_length)
    x_test = sequence.pad_sequences(x_test, model.config.seq_length)

    # y_train = np_utils.to_categorical(y_train,)
    # y_test = np_utils.to_categorical(y_test,)
    #get 多标签分类
    y_train = get_multilabel_QiXueJinYe_Multiclass(y_train)
    y_test = get_multilabel_QiXueJinYe_Multiclass(y_test)

    #create session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    #writer.add_graph(session.graph)
    writer = tf.summary.FileWriter(tensorboard_dir, session.graph)
    print("Training and evaluate")
    total_batch = 0
    best_acc_val = 0.0
    last_improved =0
    require_improvement = 1000
    flag = False
    #print('x', model.input_x[0, :].eval(feed_dict=feed_dict_eval))
    for epoch in range(model.config.num_epochs):
        print('Epoch',epoch)
        batch_train = batch_iter(x_train, y_train, model.config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.keep_prob: model.config.dropout_keep_prob
            }
            if total_batch % model.config.save_per_batch:
                s = session.run(merged_summary, feed_dict =feed_dict)
                writer.add_summary(s, total_batch)
                # saver.save(sess=session, save_path=save_dir)

            if total_batch % model.config.print_per_batch==0:
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc],feed_dict= feed_dict)
                #?
                loss_val, acc_val = evaluate(model,session, x_test, y_test,)
                # b=tf.summary.histogram('Loss',loss_val)
                # a=session.run(b)
                # writer.add_summary(b, total_batch)


                best_acc_val = acc_val
                last_improved = total_batch
                saver.save(sess=session, save_path=save_dir)
                improved_str = '*'
                # else:
                #     improved_str = ''

                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, '
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, improved_str))
            # list_gradient = session.run([model.compute[0]], feed_dict=feed_dict)#元组列表
            # temp = session.run([model.embedding])
            # print(temp, 'shape', )
            # values, index =list_gradient[0]#每个元组有gradient，var
            # print(list_gradient[0][0][0], np.shape(list_gradient[0][0][0]))
            # print(list_gradient[0][0][1])
            # print('0','values', np.shape(values), 'index', np.shape(index))
            session.run(model.optim, feed_dict = feed_dict)
            total_batch += 1
    print('----------------Predicting---------------')
    y_label = session.run(model.y_pred_cls, feed_dict={
        model.input_x : x_test,
        model.keep_prob : 1.0
    })
    # print('label', y_label)
    # print(type(y_label))
    # accuarcy = accuarcy_xsh(y_label,y_test,model.config.num_classes)
    # print('accuarcy', accuarcy)
    # score1 = f1_score(y_label,y_test,average='weighted')
    # score2 = f1_score(y_label, y_test,average='samples')
    # score3 = f1_score(y_label, y_test, average='micro')
    # score4 = f1_score(y_label, y_test,average='macro')
    # print('Score_weighted', score1)
    # print('Score_samples', score2)
    # print('Score_micro',score3)
    # print('Score_macro',score4)
    session.close()
def train3(model, save_dir,cidian_dir):
    #八纲
    print("config Tensorboard and saver")

    tensorboard_dir = 'C:/logs'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy",model.acc)

    #merge all
    merged_summary = tf.summary.merge_all()


    #config saver
    saver = tf.train.Saver()
    # save_dir = './checkpoints/QiXueJinYe/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("loading data")
    #input data
    #the name of the dictionary
    file_cidian = open(cidian_dir, 'rb', )
    indexword = pickle.load(file_cidian)
    vectorword = pickle.load(file_cidian)

    #the directory of the input data
    x_train, y_train = data_build_label('./data/bingli_exp_result/train')
    x_test, y_test = data_build_label('./data/bingli_exp_result/test')
    #convert y to multi_label

    # print('Y-train', y_train,type(y_train))
    # print('Y_test',y_test)
    y_temp = []
    count = 0

    #converse word to index
    x_train = word_to_index(x_train, indexword)
    x_test = word_to_index(x_test, indexword)
    # print('x_train', x_train)

    #padding the sequence
    x_train = sequence.pad_sequences(x_train, model.config.seq_length)
    x_test = sequence.pad_sequences(x_test, model.config.seq_length)

    # y_train = np_utils.to_categorical(y_train,)
    # y_test = np_utils.to_categorical(y_test,)
    #get 多标签分类
    y_train = get_multilabel_BaGang_Multiclass(y_train)
    y_test = get_multilabel_BaGang_Multiclass(y_test)

    #create session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    #writer.add_graph(session.graph)
    writer = tf.summary.FileWriter(tensorboard_dir, session.graph)
    print("Training and evaluate")
    total_batch = 0
    #print('x', model.input_x[0, :].eval(feed_dict=feed_dict_eval))
    for epoch in range(model.config.num_epochs):
        print('Epoch',epoch)
        batch_train = batch_iter(x_train, y_train, model.config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.keep_prob: model.config.dropout_keep_prob
            }
            if total_batch % model.config.save_per_batch:
                s = session.run(merged_summary, feed_dict =feed_dict)
                writer.add_summary(s, total_batch)
                # saver.save(sess=session, save_path=save_dir)

            if total_batch % model.config.print_per_batch==0:
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc],feed_dict= feed_dict)
                #?
                loss_val, acc_val = evaluate(model, session, x_test, y_test)
                # b=tf.summary.histogram('Loss',loss_val)
                # a=session.run(b)
                # writer.add_summary(b, total_batch)


                best_acc_val = acc_val
                last_improved = total_batch
                saver.save(sess=session, save_path=save_dir)
                improved_str = '*'
                # else:
                #     improved_str = ''

                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, '
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, improved_str))
                print()
            # list_gradient = session.run([model.compute[0]], feed_dict=feed_dict)#元组列表
            # temp = session.run([model.embedding])
            # print(temp, 'shape', )
            # values, index =list_gradient[0]#每个元组有gradient，var
            # print(list_gradient[0][0][0], np.shape(list_gradient[0][0][0]))
            # print(list_gradient[0][0][1])
            # print('0','values', np.shape(values), 'index', np.shape(index))
            session.run(model.optim, feed_dict = feed_dict)
            total_batch += 1
    print('----------------Predicting---------------')
    # y_label = session.run(model.y_pred_cls_3, feed_dict={
    #     model.input_x : x_test,
    #     model.keep_prob : 1.0
    # })
    # print('label', y_label)
    # print(type(y_label))
    # accuarcy = accuarcy_xsh(y_label,y_test,model.config.num_classes)
    # print('accuarcy', accuarcy)
    # score1 = f1_score(y_label,y_test,average='weighted')
    # score2 = f1_score(y_label, y_test,average='samples')
    # score3 = f1_score(y_label, y_test, average='micro')
    # score4 = f1_score(y_label, y_test,average='macro')
    # print('Score_weighted', score1)
    # print('Score_samples', score2)
    # print('Score_micro',score3)
    # print('Score_macro',score4)
    session.close()

def train4(model, save_dir,cidian_dir):
        #卫气营血
        print("config Tensorboard and saver")

        tensorboard_dir = 'C:/logs'
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        tf.summary.scalar("loss", model.loss)
        tf.summary.scalar("accuracy", model.acc)

        # merge all
        merged_summary = tf.summary.merge_all()

        # config saver
        saver = tf.train.Saver()
        # save_dir = './checkpoints/Test/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("loading data")
        # input data
        # the name of the dictionary
        file_cidian = open(cidian_dir, 'rb', )
        indexword = pickle.load(file_cidian)
        vectorword = pickle.load(file_cidian)

        # the directory of the input data
        x_train, y_train = data_build_label('./data/bingli_exp_result/train')
        x_test, y_test = data_build_label('./data/bingli_exp_result/test')
        # convert y to multi_label

        print('Y-train', y_train,type(y_train))
        print('Y_test',y_test)
        y_temp = []
        count = 0

        # converse word to index
        x_train = word_to_index(x_train, indexword)
        x_test = word_to_index(x_test, indexword)
        # print('x_train', x_train)

        # padding the sequence
        x_train = sequence.pad_sequences(x_train, model.config.seq_length)
        x_test = sequence.pad_sequences(x_test, model.config.seq_length)

        # y_train = np_utils.to_categorical(y_train,)
        # y_test = np_utils.to_categorical(y_test,)
        # get 多标签分类
        y_train = get_multilabel_WeiQiYingXue_Multiclass(y_train)
        y_test = get_multilabel_WeiQiYingXue_Multiclass(y_test)

        # create session
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        # writer.add_graph(session.graph)
        writer = tf.summary.FileWriter(tensorboard_dir, session.graph)
        print("Training and evaluate")
        total_batch = 0
        best_acc_val = 0.0
        last_improved = 0
        require_improvement = 1000
        flag = False

        feed_dict_eval = {
            model.input_x: x_train,
            model.input_y: y_train,
            model.keep_prob: model.config.dropout_keep_prob
        }

        # print('x', model.input_x[0, :].eval(feed_dict=feed_dict_eval))
        for epoch in range(model.config.num_epochs):
            print('Epoch', epoch)
            batch_train = batch_iter(x_train, y_train, model.config.batch_size)
            for x_batch, y_batch in batch_train:
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.keep_prob: model.config.dropout_keep_prob
                }
                if total_batch % model.config.save_per_batch:
                    s = session.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(s, total_batch)
                    # saver.save(sess=session, save_path=save_dir)

                if total_batch % model.config.print_per_batch == 0:
                    feed_dict[model.keep_prob] = 1.0
                    loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                    # ?
                    loss_val, acc_val = evaluate(model, session, x_test, y_test)
                    # b=tf.summary.histogram('Loss',loss_val)
                    # a=session.run(b)
                    # writer.add_summary(b, total_batch)


                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_dir)
                    improved_str = '*'
                    # else:
                    #     improved_str = ''

                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, '
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, improved_str))
                    print()
                # list_gradient = session.run([model.compute[0]], feed_dict=feed_dict)#元组列表
                # temp = session.run([model.embedding])
                # print(temp, 'shape', )
                # values, index =list_gradient[0]#每个元组有gradient，var
                # print(list_gradient[0][0][0], np.shape(list_gradient[0][0][0]))
                # print(list_gradient[0][0][1])
                # print('0','values', np.shape(values), 'index', np.shape(index))
                session.run(model.optim, feed_dict=feed_dict)
                total_batch += 1
        print('----------------Predicting---------------')
        # y_label = session.run(model.y_pred_cls_3, feed_dict={
        #     model.input_x: x_test,
        #     model.keep_prob: 1.0
        # })
        # print('label', y_label)
        # print(type(y_label))
        # accuarcy = accuarcy_xsh(y_label, y_test, model.config.num_classes)
        # print('accuarcy', accuarcy)
        # score1 = f1_score(y_label, y_test, average='weighted')
        # score2 = f1_score(y_label, y_test, average='samples')
        # score3 = f1_score(y_label, y_test, average='micro')
        # score4 = f1_score(y_label, y_test, average='macro')
        # print('Score_weighted', score1)
        # print('Score_samples', score2)
        # print('Score_micro', score3)
        # print('Score_macro', score4)
        session.close()
def train5(model, save_dir,cidian_dir):
    #脏腑
    #save_dir 保存模型的路径
    print("config Tensorboard and saver")

    tensorboard_dir = 'C:/logs'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy",model.acc)

    #merge all
    merged_summary = tf.summary.merge_all()


    #config saver
    saver = tf.train.Saver()
    # save_dir = './checkpoints/zangfu/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("loading data")
    #input data
    #the name of the dictionary
    file_cidian = open(cidian_dir, 'rb', )
    indexword = pickle.load(file_cidian)
    vectorword = pickle.load(file_cidian)

    #the directory of the input data
    x_train, y_train = data_build_label('./data/bingli_exp_result/train')
    x_test, y_test = data_build_label('./data/bingli_exp_result/test')
    print('X-test',x_test)
    #convert y to multi_label

    # print('Y-train', y_train,type(y_train))
    # print('Y_test',y_test)
    y_temp = []
    count = 0

    #converse word to index
    print('x_test', x_test[1])
    x_train = word_to_index(x_train, indexword)
    x_test = word_to_index(x_test, indexword)
    # print('x_train', x_train)

    #padding the sequence
    x_train = sequence.pad_sequences(x_train, model.config.seq_length)
    x_test = sequence.pad_sequences(x_test, model.config.seq_length)

    # y_train = np_utils.to_categorical(y_train,)
    # y_test = np_utils.to_categorical(y_test,)
    #get 多标签分类，这里是不同的train函数的主要区别
    y_train = get_multilabel_SanJiao_Multiclass(y_train)
    y_test = get_multilabel_SanJiao_Multiclass(y_test)

    #create session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    #writer.add_graph(session.graph)
    writer = tf.summary.FileWriter(tensorboard_dir, session.graph)
    print("Training and evaluate")
    total_batch = 0
    best_acc_val = 0.0
    last_improved =0
    require_improvement = 1000
    flag = False

    # feed_dict_eval = {
    #     model.input_x: x_train,
    #     model.input_y: y_train,
    #     model.keep_prob: model.config.dropout_keep_prob
    # }


    #print('x', model.input_x[0, :].eval(feed_dict=feed_dict_eval))
    for epoch in range(model.config.num_epochs):
        print('Epoch',epoch)
        batch_train = batch_iter(x_train, y_train, model.config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.keep_prob: model.config.dropout_keep_prob
            }
            if total_batch % model.config.save_per_batch:
                s = session.run(merged_summary, feed_dict =feed_dict)
                writer.add_summary(s, total_batch)
                # saver.save(sess=session, save_path=save_dir)

            if total_batch % model.config.print_per_batch==0:
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc],feed_dict= feed_dict)
                #?
                loss_val, acc_val = evaluate(model, session, x_test, y_test)
                # b=tf.summary.histogram('Loss',loss_val)
                # a=session.run(b)
                # writer.add_summary(b, total_batch)


                best_acc_val = acc_val
                last_improved = total_batch
                saver.save(sess=session, save_path=save_dir)
                improved_str = '*'
                # else:
                #     improved_str = ''

                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, '
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, improved_str))
                print()
            # list_gradient = session.run([model.compute[0]], feed_dict=feed_dict)#元组列表
            # temp = session.run([model.embedding])
            # print(temp, 'shape', )
            # values, index =list_gradient[0]#每个元组有gradient，var
            # print(list_gradient[0][0][0], np.shape(list_gradient[0][0][0]))
            # print(list_gradient[0][0][1])
            # print('0','values', np.shape(values), 'index', np.shape(index))
            session.run(model.optim, feed_dict = feed_dict)
            total_batch += 1
    print('----------------Predicting---------------')
    # y_label = session.run(model.y_pred_cls_3, feed_dict={
    #     model.input_x : x_test,
    #     model.keep_prob : 1.0
    # })
    # print('label', y_label)
    # print(type(y_label), np.shape(y_label))
    # y_str = y_label.astype(int)
    # print(y_str)
    # # 不同的train函数需要修改
    # y_str = get_inverse_multilabel_SanJiao(y_str)
    # print(y_str)
    # accuarcy = accuarcy_xsh(y_label,y_test,model.config.num_classes)
    # print('accuarcy', accuarcy)
    # score1 = f1_score(y_label,y_test,average='weighted')
    # score2 = f1_score(y_label, y_test,average='samples')
    # score3 = f1_score(y_label, y_test, average='micro')
    # score4 = f1_score(y_label, y_test,average='macro')
    # print('Score_weighted', score1)
    # print('Score_samples', score2)
    # print('Score_micro',score3)
    # print('Score_macro',score4)
    session.close()
def test1(model, save_dir,cidian_dir):
    #脏腑
    #save_dir 保存模型的路径
    print("config Tensorboard and saver")


    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy",model.acc)

    #merge all
    merged_summary = tf.summary.merge_all()


    #config saver
    saver = tf.train.Saver()
    # save_dir = './checkpoints/zangfu/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("loading data")
    #input data
    #the name of the dictionary
    file_cidian = open(cidian_dir, 'rb', )
    indexword = pickle.load(file_cidian)
    vectorword = pickle.load(file_cidian)

    #the directory of the input data
    x_test, y_test = data_build_label('./data/bingli_exp_result/test')
    print('y_test',y_test)
    #convert y to multi_label

    #converse word to index
    y_return = y_test
    x_test = word_to_index(x_test, indexword)
    # print('x_train', x_train)

    #padding the sequence
    x_test = sequence.pad_sequences(x_test, model.config.seq_length)

    # y_train = np_utils.to_categorical(y_train,)
    # y_test = np_utils.to_categorical(y_test,)
    #get 多标签分类，这里是不同的train函数的主要区别
    y_test = get_multilabel_ZangFu_Multiclass(y_test)
    #create session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    #writer.add_graph(session.graph)
    print("Training and evaluate")
    total_batch = 0
    best_acc_val = 0.0
    last_improved =0
    require_improvement = 1000
    flag = False
    saver.restore(session,save_dir)

    print('----------------Predicting---------------')
    #2018年12月11日修改
    y_label = session.run(model.y_pred_cls, feed_dict={
        model.input_x: x_test,
        model.keep_prob: 1.0
    })
    # print('Y',y_label)
    ZangFu_multiclass = './data/ZangFu_multiclass.pkl'
    y_str = []
    with open(ZangFu_multiclass, 'rb') as f:
        ZangFu_label = pickle.load(f)
    for item in y_label:
        y_str.append(ZangFu_label[item])
    # y_label = session.run(model.y_pred_cls_3, feed_dict={
    #     model.input_x : x_test,
    #     model.keep_prob : 1.0
    # })
    # print('label', y_label)
    # print(type(y_label), np.shape(y_label))
    # y_str = y_label.astype(int)
    # y_str = get_inverse_multilabel_ZangFu(y_str)
    # print(y_str)
    # accuarcy = accuarcy_xsh(y_label,y_test,model.config.num_classes)
    # print('accuarcy', accuarcy)
    # score1 = f1_score(y_label,y_test,average='weighted')
    # score2 = f1_score(y_label, y_test,average='samples')
    # score3 = f1_score(y_label, y_test, average='micro')
    # score4 = f1_score(y_label, y_test,average='macro')
    # print('Score_weighted', score1)
    # print('Score_samples', score2)
    # print('Score_micro',score3)
    # print('Score_macro',score4)
    session.close()
    return y_str
    # return y_str, y_return
def test2(model, save_dir,cidian_dir):
    #气血津液
    #save_dir 保存模型的路径
    print("config Tensorboard and saver")


    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy",model.acc)

    #merge all
    merged_summary = tf.summary.merge_all()


    #config saver
    saver = tf.train.Saver()
    # save_dir = './checkpoints/zangfu/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("loading data")
    #input data
    #the name of the dictionary
    file_cidian = open(cidian_dir, 'rb', )
    indexword = pickle.load(file_cidian)
    vectorword = pickle.load(file_cidian)

    #the directory of the input data
    x_test, y_test = data_build_label('./data/bingli_exp_result/test')
    #convert y to multi_label

    #converse word to index

    x_test = word_to_index(x_test, indexword)
    # print('x_train', x_train)

    #padding the sequence
    x_test = sequence.pad_sequences(x_test, model.config.seq_length)

    # y_train = np_utils.to_categorical(y_train,)
    # y_test = np_utils.to_categorical(y_test,)
    #get 多标签分类，这里是不同的train函数的主要区别
    y_test = get_multilabel_QiXueJinYe_Multiclass(y_test)
    #create session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    #writer.add_graph(session.graph)
    print("Training and evaluate")
    total_batch = 0
    best_acc_val = 0.0
    last_improved =0
    require_improvement = 1000
    flag = False
    saver.restore(session,save_dir)

    print('----------------Predicting---------------')
    y_label = session.run(model.y_pred_cls, feed_dict={
        model.input_x: x_test,
        model.keep_prob: 1.0
    })
    # print('Y',y_label)
    ZangFu_multiclass = './data/QiXueJinYe_multiclass.pkl'
    y_str = []
    with open(ZangFu_multiclass, 'rb') as f:
        ZangFu_label = pickle.load(f)
    for item in y_label:
        y_str.append(ZangFu_label[item])
    session.close()
    return y_str
def test3(model, save_dir,cidian_dir):
    #八纲
    #save_dir 保存模型的路径
    print("config Tensorboard and saver")

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy",model.acc)

    #merge all
    merged_summary = tf.summary.merge_all()


    #config saver
    saver = tf.train.Saver()
    # save_dir = './checkpoints/zangfu/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("loading data")
    #input data
    #the name of the dictionary
    file_cidian = open(cidian_dir, 'rb', )
    indexword = pickle.load(file_cidian)
    vectorword = pickle.load(file_cidian)

    #the directory of the input data
    x_test, y_test = data_build_label('./data/bingli_exp_result/test')
    #convert y to multi_label

    #converse word to index

    x_test = word_to_index(x_test, indexword)
    # print('x_train', x_train)

    #padding the sequence
    x_test = sequence.pad_sequences(x_test, model.config.seq_length)

    # y_train = np_utils.to_categorical(y_train,)
    # y_test = np_utils.to_categorical(y_test,)
    #get 多标签分类，这里是不同的train函数的主要区别
    y_test = get_multilabel_BaGang_Multiclass(y_test)
    #create session

    session = tf.Session()

    session.run(tf.global_variables_initializer())

    #writer.add_graph(session.graph)
    print("Training and evaluate")
    total_batch = 0
    best_acc_val = 0.0
    last_improved =0
    require_improvement = 1000
    flag = False
    saver.restore(session,save_dir)

    print('----------------Predicting---------------')
    y_label = session.run(model.y_pred_cls, feed_dict={
        model.input_x: x_test,
        model.keep_prob: 1.0
    })
    # print('Y',y_label)
    ZangFu_multiclass = './data/BaGang_multiclass.pkl'
    y_str = []
    with open(ZangFu_multiclass, 'rb') as f:
        ZangFu_label = pickle.load(f)
    for item in y_label:
        y_str.append(ZangFu_label[item])
    session.close()
    return y_str

def test4(model, save_dir,cidian_dir):
    #脏腑
    #save_dir 保存模型的路径
    print("config Tensorboard and saver")


    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy",model.acc)

    #merge all
    merged_summary = tf.summary.merge_all()


    #config saver
    saver = tf.train.Saver()
    # save_dir = './checkpoints/zangfu/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("loading data")
    #input data
    #the name of the dictionary
    file_cidian = open(cidian_dir, 'rb', )
    indexword = pickle.load(file_cidian)
    vectorword = pickle.load(file_cidian)

    #the directory of the input data
    x_test, y_test = data_build_label('./data/bingli_exp_result/test')
    #convert y to multi_label

    #converse word to index

    x_test = word_to_index(x_test, indexword)
    # print('x_train', x_train)

    #padding the sequence
    x_test = sequence.pad_sequences(x_test, model.config.seq_length)

    # y_train = np_utils.to_categorical(y_train,)
    # y_test = np_utils.to_categorical(y_test,)
    #get 多标签分类，这里是不同的train函数的主要区别
    y_test = get_multilabel_WeiQiYingXue_Multiclass(y_test)
    #create session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    #writer.add_graph(session.graph)
    print("Training and evaluate")
    total_batch = 0
    best_acc_val = 0.0
    last_improved =0
    require_improvement = 1000
    flag = False
    saver.restore(session,save_dir)

    print('----------------Predicting---------------')
    y_label = session.run(model.y_pred_cls, feed_dict={
        model.input_x: x_test,
        model.keep_prob: 1.0
    })
    # print('Y',y_label)
    ZangFu_multiclass = './data/WeiQiYingXue_multiclass.pkl'
    y_str = []
    with open(ZangFu_multiclass, 'rb') as f:
        ZangFu_label = pickle.load(f)
    for item in y_label:
        y_str.append(ZangFu_label[item])
    session.close()
    return y_str
def test5(model, save_dir,cidian_dir):
    #脏腑
    #save_dir 保存模型的路径
    print("config Tensorboard and saver")


    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy",model.acc)

    #merge all
    merged_summary = tf.summary.merge_all()


    #config saver
    saver = tf.train.Saver()
    # save_dir = './checkpoints/zangfu/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("loading data")
    #input data
    #the name of the dictionary
    file_cidian = open(cidian_dir, 'rb', )
    indexword = pickle.load(file_cidian)
    vectorword = pickle.load(file_cidian)

    #the directory of the input data
    x_test, y_test = data_build_label('./data/bingli_exp_result/test')
    #convert y to multi_label

    #converse word to index

    x_test = word_to_index(x_test, indexword)
    # print('x_train', x_train)

    #padding the sequence
    x_test = sequence.pad_sequences(x_test, model.config.seq_length)

    # y_train = np_utils.to_categorical(y_train,)
    # y_test = np_utils.to_categorical(y_test,)
    #get 多标签分类，这里是不同的train函数的主要区别
    y_test = get_multilabel_SanJiao_Multiclass(y_test)
    #create session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    #writer.add_graph(session.graph)
    print("Training and evaluate")
    total_batch = 0
    best_acc_val = 0.0
    last_improved =0
    require_improvement = 1000
    flag = False
    saver.restore(session,save_dir)

    print('----------------Predicting---------------')
    y_label = session.run(model.y_pred_cls, feed_dict={
        model.input_x: x_test,
        model.keep_prob: 1.0
    })
    # print('Y',y_label)
    ZangFu_multiclass = './data/SanJiao_multiclass.pkl'
    y_str = []
    with open(ZangFu_multiclass, 'rb') as f:
        ZangFu_label = pickle.load(f)
    for item in y_label:
        y_str.append(ZangFu_label[item])
    session.close()
    return y_str







if __name__ =='__main__':
    print("config CNN")
    config = Config(10,13)
    model = TextCnn(config)
    save_dir = './checkpoints/Test'
    train2(model, save_dir)
    y_str =test2(model, save_dir)
    print(y_str)
