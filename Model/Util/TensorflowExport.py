# -*- coding: utf-8 -*- 
# @author : zyh  @time :2018/10/29
import numpy as np
import os
from Model.Util import config
class TensorflowExport:
    def save_content(self, np_array, var_name,modelName):
        save_dir = config.saveModelDir
        content_file = "{}/{}.csv".format(save_dir+"_"+modelName, var_name)
        shape_file = "{}/{}.shape".format(save_dir+"_"+modelName, var_name)
        if not os.path.exists(save_dir+"_"+modelName):
            os.mkdir(save_dir+"_"+modelName)
        np.savetxt(shape_file, np.asarray(np_array.shape), fmt="%i")
        np.savetxt(content_file, np.ndarray.flatten(np_array), fmt="%10.8f")
