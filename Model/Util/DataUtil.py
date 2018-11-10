# -*- coding: utf-8 -*-

import pandas as pd
from Model.Util import config
from sklearn.model_selection import StratifiedKFold
import numpy as np
from Model.Util import config
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
from Model.Util.TensorflowExport import TensorflowExport
class DataUtil:

    def __init__(self):
        self.count=0


    def load_data(self):
        dftrain = pd.read_csv(config.TRAIN_FILE)
        dftest = pd.read_csv(config.TEST_FILE)

        cols = [col for col in dftrain.columns if col not in config.IGNORE_COLS]

        X_train = dftrain[cols].values  #ndarray (10000,38)
        X_test = dftest[cols].values  #ndarray (10000,38)

        y_train = dftrain['target'].values
        ids_test=dftest['id'].values

        folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                                     random_state=config.RANDOM_SEED).split(X_train, y_train))

        return dftrain,dftest,X_train,y_train,X_test,ids_test,folds

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size

        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

        # shuffle three lists simutaneously

    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def predict(self, model,Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: loss and accuracy
        """
        # dummy y
        y = np.array(y).reshape(-1, 1)
        feed_dict = {model.feat_index: Xi,
                     model.feat_value: Xv,
                     model.label: y,
                     # model.dropout_keep_deep: [1.0] * len(model.dropout_dep),
                     # model.train_phase: True
                     # model.train_phase:1
                     }

        _, loss, pred = model.sess.run([model.optimizer, model.loss, model.out], feed_dict=feed_dict)
        pred = np.array(pred).reshape(-1, )

        accuracy = accuracy_score(np.array(y).reshape(-1, ), np.where(pred > 0.5, 1, 0))
        return loss, accuracy

    def fit_on_batch(self, model,Xi, Xv, y):

        feed_dict = {model.feat_index: Xi,
                     model.feat_value: Xv,
                     model.label: y,
                     # model.dropout_keep_deep: model.dropout_dep,
                     # model.train_phase: 1

                     }

        loss, opt = model.sess.run([model.loss, model.optimizer], feed_dict=feed_dict)

        return loss

    def fit(self, model,fold_index, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None):
        '''
        training validation
        '''
        for epoch in range(model.epoch):
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / model.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, model.batch_size, i)
                self.fit_on_batch(model,Xi_batch, Xv_batch, y_batch)
            loss, accuracy = self.predict(model,Xi_valid, Xv_valid, y_valid)
            print("foldIndex", fold_index, "epoch", epoch, "loss", loss, 'accuracy', accuracy)

    def get_batch_withouLabel(self, Xi_test, Xv_test, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size

        end = end if end < len(Xi_test) else len(Xi_test)
        return Xi_test[start:end], Xv_test[start:end]

