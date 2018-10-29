# -*- coding: utf-8 -*- 
# @author : zyh  @time :2018/10/29
from Model.Util import config
import tensorflow as tf
pnn_params = {
    "embedding_size":8,
    "deep_layers":[32,32],
    "dropout_deep":[0.5,0.5,0.5],
    "deep_layer_activation":tf.nn.relu,
    "epoch":30,
    "batch_size":1024,
    "learning_rate":0.001,
    "optimizer":"adam",
    "batch_norm":1,
    "batch_norm_decay":0.995,
    "verbose":True,
    "random_seed":config.RANDOM_SEED,
    "deep_init_size":50,
    "firstLayer":"layer_0",
    "firstBias":"bias_0"
   }