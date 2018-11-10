# -*- coding: utf-8 -*- 
# @author : zyh  @time :2018/10/30
import numpy as np
import tensorflow as tf
from Model.Util import DataUtil
from Model.Util import config
from Model.Util.DataReader import FeatureDictionary, DataParser
from Model.Util.TensorflowExport import TensorflowExport
from Model.DeepFMModel import DeepFMParams

class DeepFM:
    def __init__(self, feature_size, field_size,
                 embedding_size,
                 deep_layers,
                 dropout_deep,
                 epoch, batch_size,
                 learning_rate, optimizer,
                 batch_norm, batch_norm_decay,
                 verbose, random_seed):
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.epoch = epoch

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(config.RANDOM_SEED)

            # build weights
            self.FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
            self.FM_W = tf.get_variable(name='fm_firstOrder', shape=[self.feature_size, 1],
                                        initializer=tf.glorot_normal_initializer())
            self.FM_V = tf.get_variable(name='fm_secondOrder', shape=[self.feature_size, self.embedding_size],
                                        initializer=tf.glorot_normal_initializer())

            # build features
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')  # N * F
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')  # N * F
            self.label = tf.placeholder("float", shape=[None, 1])
            # self.train_phase=tf.placeholder(tf.int32,shape=[None],name='train_phase')

            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])  # N*F*1



            with tf.variable_scope("firstOrder"):
                self.y_first_order = tf.nn.embedding_lookup(self.FM_W, self.feat_index)  # N * F * 1
                self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)
                self.y_first_order = tf.reduce_sum(self.y_first_order, 1, keepdims=True)  # N * 1

            with tf.variable_scope("secondOrder"):
                self.embeddings = tf.nn.embedding_lookup(self.FM_V, self.feat_index)  # N * F * K
                self.embeddings = tf.multiply(self.embeddings, feat_value)  # N * F * K

                self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * K
                self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

                self.squared_features_emb = tf.square(self.embeddings)
                self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K
                self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square,
                                                        self.squared_sum_features_emb)  # None * K

            deepInputs = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])
            with tf.variable_scope("deepPart"):
                for i in range(len(self.deep_layers)):
                    deepInputs = tf.contrib.layers.fully_connected(inputs=deepInputs, num_outputs=self.deep_layers[i],
                                                                   weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                                       0.0001), scope='mlp%d' % i)
                    # todo
                    tf.nn.dropout(deepInputs, keep_prob=self.dropout_deep[i])

            self.y_deep = tf.contrib.layers.fully_connected(inputs=deepInputs, num_outputs=1, activation_fn=tf.identity,
                                                            weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                                0.0001))
            self.y_deep = tf.reshape(self.y_deep, shape=[-1, 1])  # N * 1

            self.bias = tf.ones_like(self.y_deep, dtype=tf.float32)

            self.out = tf.add_n([self.y_first_order, self.y_deep, self.bias],
                                )

            self.out = tf.nn.sigmoid(self.out, name='output')
            self.loss = tf.losses.log_loss(self.label, self.out)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)


if __name__ == '__main__':
    data_util = DataUtil.DataUtil()

    dfTrain, dfTest, X_train, y_train, X_test, ids_test, folds = data_util.load_data()
    fd = FeatureDictionary(dfTrain=dfTrain,
                           dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS, ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)

    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)
    DeepFMParams.deepfm_params['feature_size'] = fd.feat_dim
    DeepFMParams.deepfm_params['field_size'] = len(Xi_train[0])
    _get = lambda x, l: [x[i] for i in l]

    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)
        deepfm = DeepFM(**DeepFMParams.deepfm_params)
        data_util.fit(deepfm, i, Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)
        # 第一个fold导出pb文件
        if i == 0:
            constant_graph = tf.graph_util.convert_variables_to_constants(deepfm.sess, deepfm.sess.graph_def,
                                                                          ["output"])
            with tf.gfile.FastGFile("frozen_model_deepfm.pb", mode='wb') as f:
                f.write(constant_graph.SerializeToString())

            export = TensorflowExport()

            Xi_batch, Xv_batch = data_util.get_batch_withouLabel(Xi_test, Xv_test, deepfm.batch_size, 0)
            export.save_content(np.array(Xi_batch), 'feat_index', 'deepfm')
            export.save_content(np.array(Xv_batch), 'feat_value', 'deepfm')

            # export.save_content(np.array(y_valid_).reshape(-1,1), 'label', 'nfm')
            out_eval = deepfm.out.eval({deepfm.feat_index: Xi_test, deepfm.feat_value: Xv_test}, session=deepfm.sess)
            export.save_content(np.array(out_eval), 'predition', 'deepfm')
            # 为了测试导出pb文件给dl4j使用 第一个fold暂停
            break

