import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util

root_A = "C:/Users/gpuuser1/Downloads/our_model/our_net1/Dataset/trainsets/MSRS/vi"
root_saveA = "C:/Users/gpuuser1/Downloads/our_model/our_net1/Dataset/trainsets/MSRS/VI_Y/"
paths_A,names_A = util.get_image_path_name(root_A)

for index in range(len(paths_A)):
    A_path = root_A+'/'+paths_A[index]
    #print(A_path)
    img_A = util.convert_rgb2y(A_path)
    save_path = root_saveA+paths_A[index]
    print(save_path)
    util.imwrite(img_A,save_path)

