# -*- coding: utf-8 -*- 
# @author : zyh  @time :2018/10/30
from Model.Util import config
deepfm_params = {
    "embedding_size":8,
    "deep_layers":[32,32],
    "dropout_deep":[0.5,0.5,0.5],
    "epoch":30,
    "batch_size":1024,
    "learning_rate":0.001,
    "optimizer":"adam",
    "batch_norm":1,
    "batch_norm_decay":0.995,
    "verbose":True,
    "random_seed":config.RANDOM_SEED,
}




