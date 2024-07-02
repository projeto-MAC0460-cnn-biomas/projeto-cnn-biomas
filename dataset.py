import numpy as np
from array_tools import *
from random import sample
from PIL import Image
from os import walk

gs = np.vectorize(lambda x: rgb_gray(*x))

cv_table = {}

def initialize_dataset(path, limit=None):
    test_fl = next(walk(path + "/seg_test"))[2]
    train_fl = next(walk(path + "/seg_train"))[2]
    ds = []
    
    
def nominal_load(path, cv_table=None, wl=None, limit=None):
    cl = next(walk(path))[2]
    
    if wl is not None:
        cl = list(filter(lambda x: x in wl, cl))
    
    ds = []
    
    for i in cl:
        fl = next(walk(path))[2]
        cds = []
        
        for j in next(walk(path + "/" + i))[2]:
        
            if limit is not None:
                if len(cds) >= limit:
                    break
            
            if cv_table is not None:
                ct = cv_table[i]
            else:
                ct = i
            cds.append((path + "/" + i + "/" + j, ct))
    
        ds = ds + cds
    return ds

def load_img(path):
    return gs(np.array(Image.open(path))).tolist()
