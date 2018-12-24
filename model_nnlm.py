<<<<<<< HEAD
import pickle
import numpy as np
import tensorflow as tf
import numpy as np

class Config():
    def __init__(self,number,classes):
        self.num_epochs =number
        self.num_classes = classes
    embedding_dim = 128
    seq_length = 100
    # num_classes = 7
    num_filters = 128
    kernel_size = 3
    vocab_size = 5000

    hidden_dim = 256

    dropout_keep_prob = 0.5
    learning_rate = 1e-3

    batch_size =64
    # num_epochs = number

    print_per_batch =10
    save_per_batch =100

class TextCnn():
    def __init__(self, config):
        self.config = config
        # input data
        self.input_x =tf.placeholder(tf.int32,[None, self.config.seq_length],name = 'input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name = 'input_y')
        self.keep_prob = tf.placeholder(tf.float32, name ='keep_prob')
        self.cnn()
    def cnn(self):

        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim],)
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
        with tf.name_scope("cnn"):
            # conv = tf.layers.conv1d(self.embedding_inputs, self.config.num_filters,self.config.kernel_size,padding='valid',  name ='conv')
            #
            # print('conv',conv.shape)

            gmp = tf.reduce_max(self.embedding_inputs, reduction_indices=[1], name='gmp')
            # print('gmp', gmp.shape)
        with tf.name_scope("score"):
            fc = tf.layers.dense(inputs=gmp, units=self.config.hidden_dim, name='fcl')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            # print('fc', np.shape(fc))
            # fc_1 = tf.layers.dense(fc, 64, name ='fc_22')
            # fc_1 = tf.contrib.layers.dropout(fc_1,self.keep_prob)
            # fc_1 = tf.nn.relu(fc_1)
            fc2 = tf.layers.dense(fc, self.config.num_classes, name='fc-2')
            # self.logits = tf.nn.relu(fc2)
            self.logits = fc2
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits,), 1)
            # self.y_pred_cls_1 = tf.nn.sigmoid(self.logits, )
            # self.y_pred_cls_2 = self.y_pred_cls_1 > 0.5
            # self.y_pred_cls_3 = tf.cast(self.y_pred_cls_2, tf.float32)
            # print('fv', np.shape(self.logits))
            # print('y_pred_cls', np.shape(self.y_pred_cls))
        with tf.name_scope("optimize"):

            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits =self.logits,labels = self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # self.loss =0.5*tf.reduce_mean(tf.square(self.logits - self.input_y))


            self.optim =tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
            self.optim1 =tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            self.compute = self.optim1.compute_gradients(self.loss)
            self.apply = self.optim1.apply_gradients(self.compute)
            # print('compute', self.compute)

            # self.tidu = tf.gradients(self.loss, self.embedding_inputs)

        with tf.name_scope("accuracy"):

            correct_pred = tf.equal(tf.arg_max(self.input_y, 1),self.y_pred_cls)
=======
import pickle
import numpy as np
import tensorflow as tf
import numpy as np

class Config():
    def __init__(self,number,classes):
        self.num_epochs =number
        self.num_classes = classes
    embedding_dim = 128
    seq_length = 100
    # num_classes = 7
    num_filters = 128
    kernel_size = 3
    vocab_size = 5000

    hidden_dim = 256

    dropout_keep_prob = 0.5
    learning_rate = 1e-3

    batch_size =64
    # num_epochs = number

    print_per_batch =10
    save_per_batch =100

class TextCnn():
    def __init__(self, config):
        self.config = config
        # input data
        self.input_x =tf.placeholder(tf.int32,[None, self.config.seq_length],name = 'input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name = 'input_y')
        self.keep_prob = tf.placeholder(tf.float32, name ='keep_prob')
        self.cnn()
    def cnn(self):

        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim],)
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
        with tf.name_scope("cnn"):
            # conv = tf.layers.conv1d(self.embedding_inputs, self.config.num_filters,self.config.kernel_size,padding='valid',  name ='conv')
            #
            # print('conv',conv.shape)

            gmp = tf.reduce_max(self.embedding_inputs, reduction_indices=[1], name='gmp')
            # print('gmp', gmp.shape)
        with tf.name_scope("score"):
            fc = tf.layers.dense(inputs=gmp, units=self.config.hidden_dim, name='fcl')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            # print('fc', np.shape(fc))
            # fc_1 = tf.layers.dense(fc, 64, name ='fc_22')
            # fc_1 = tf.contrib.layers.dropout(fc_1,self.keep_prob)
            # fc_1 = tf.nn.relu(fc_1)
            fc2 = tf.layers.dense(fc, self.config.num_classes, name='fc-2')
            # self.logits = tf.nn.relu(fc2)
            self.logits = fc2
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits,), 1)
            # self.y_pred_cls_1 = tf.nn.sigmoid(self.logits, )
            # self.y_pred_cls_2 = self.y_pred_cls_1 > 0.5
            # self.y_pred_cls_3 = tf.cast(self.y_pred_cls_2, tf.float32)
            # print('fv', np.shape(self.logits))
            # print('y_pred_cls', np.shape(self.y_pred_cls))
        with tf.name_scope("optimize"):

            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits =self.logits,labels = self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # self.loss =0.5*tf.reduce_mean(tf.square(self.logits - self.input_y))


            self.optim =tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
            self.optim1 =tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            self.compute = self.optim1.compute_gradients(self.loss)
            self.apply = self.optim1.apply_gradients(self.compute)
            # print('compute', self.compute)

            # self.tidu = tf.gradients(self.loss, self.embedding_inputs)

        with tf.name_scope("accuracy"):

            correct_pred = tf.equal(tf.arg_max(self.input_y, 1),self.y_pred_cls)
>>>>>>> 0788edf81b2bbb8ef6e66d48738a23171b113672
            self.acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))