# -*- coding: utf-8 -*- 
# @author : zyh  @time :2018/10/29

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from Model.NFMModel import NFMParams
from Model.Util import DataUtil
from Model.Util import config
from Model.Util.DataReader import FeatureDictionary, DataParser
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
from Model.Util.TensorflowExport import TensorflowExport


class NFM:
    def __init__(self, feature_size, field_size,
                 embedding_size,
                 deep_layers, deep_init_size,
                 dropout_deep,
                 deep_layer_activation,
                 epoch, batch_size,
                 learning_rate, optimizer,
                 batch_norm, batch_norm_decay,
                 verbose, random_seed,
                 firstLayer, firstBias):
        self.firstBias = firstBias
        self.firstLayer = firstLayer
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        self.deep_layers = deep_layers
        self.deep_init_size = deep_init_size
        self.dropout_dep = dropout_deep
        self.deep_layers_activation = deep_layer_activation

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.train_result, self.valid_result = [], []

        self._init_graph()

    def _init_graph(self):
        '''
        N number of samples per batch
        F fields_size
        n feature_size
        k embedding_size
          build network
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_deep_deep')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

            self.weights = dict()
            self.weights['feature_embeddings'] = tf.Variable(
                tf.random_normal(shape=[self.feature_size, self.embedding_size], mean=0.0, stddev=0.001),
                name='feature_embeddings')
            # Embeddings self.feat_index ( samplesNum,fieldsNum)   self.weights['feature_embeddings'] (featureNum,embeddingNum)
            # 解释 在feat_index 10000样本 39个field  每个filed查找唯一对应特征  特征总共有feature_size个 每个特征是长度为k的向量 跟word2vec一个道理
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)  # N * F * K
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])  # N*F*1`
            self.embeddings = tf.multiply(self.embeddings, feat_value)  # N * F * K

            # first order term 一个样本 F个field 映射F个特征 每个特征对应一个数 直接累加
            self.weights['feature_bias'] = tf.Variable(tf.random_normal([self.feature_size, 1], 0.0, 1.0),
                                                       name='feature_bias')  # n * 1
            self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.feat_index)  # N * F * 1
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)  # N * F
            self.y_first_order = tf.reduce_sum(self.y_first_order, axis=1, keep_dims=True)  # N * 1

            # second order term
            # sum-square-part 一个样本 F个field 映射F个特征 每个特征对应一个k维度向量 两两做element-wise 相乘然后累加
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * K
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # squre-sum-part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square,
                                                    self.squared_sum_features_emb)  # None * K

            # Deep component
            num_layer = len(self.deep_layers)
            input_size = self.embedding_size
            glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))

            self.weights[self.firstLayer] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32
            )
            self.weights[self.firstBias] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])), dtype=np.float32
            )

            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                self.weights["layer_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                    dtype=np.float32)  # layers[i-1] * layers[i]
                self.weights["bias_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                    dtype=np.float32)  # 1 * layer[i]

            self.y_deep = self.y_second_order
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i + 1])

            # bias
            self.weights['bias'] = tf.Variable(tf.constant(0.1), name='bias')
            self.y_bias = self.weights['bias'] * tf.ones_like(self.label)

            # out
            self.out = tf.add_n([self.y_first_order, tf.reduce_sum(self.y_deep, axis=1, keep_dims=True), self.y_bias],
                                name='output')

            self.out = tf.nn.sigmoid(self.out)
            self.loss = tf.losses.log_loss(self.label, self.out)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8).minimize(self.loss)

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
    NFMParams.pnn_params['feature_size'] = fd.feat_dim
    NFMParams.pnn_params['field_size'] = len(Xi_train[0])
    _get = lambda x, l: [x[i] for i in l]

    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)
        nfm = NFM(**NFMParams.pnn_params)
        data_util.fit(nfm, i, Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)
        # 导出pb文件
        if i == (len(folds) - 1):
            constant_graph = tf.graph_util.convert_variables_to_constants(nfm.sess, nfm.sess.graph_def, ["output"])
            with tf.gfile.FastGFile("frozen_model_afm.pb", mode='wb') as f:
                f.write(constant_graph.SerializeToString())

            export = TensorflowExport()
            export.save_content(np.array(Xi_valid_), 'feat_index', 'nfm')
            export.save_content(np.array(Xv_valid_), 'feat_value', 'nfm')
            export.save_content(np.array(y_valid_), 'label', 'nfm')
