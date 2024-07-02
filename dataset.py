import numpy as np
from array_tools import *
from random import sample
from PIL import Image
from os import walk

pth = './dataset'

cv_t = {
    'buildings': 0,
    'street': 1,
    'forest': 2,
    'mountain': 3,
    'glacier': 4,
    'sea': 5
    
}

def in1(path, trlimit=None, tslimit=None):
    cv_table = {
        'buildings': 0,
        'street': 0,
        'forest': 1,
        'mountain': 1
    }
    
    wl = ['buildings', 'street', 'forest', 'mountain']
    
    trns = load_imgs(path + '/seg_train', cv_table=cv_table, wl=wl, limit=trlimit)
    tsts = load_imgs(path + '/seg_test', cv_table=cv_table, wl=wl, limit=tslimit)
    return trns, tsts
    
def in2(path, limit=None):
    cv_table = {
        'forest': 0,
        'mountain': 1
    }
    
    wl = ['forest', 'mountain']
    
    trns = load_imgs(path + '/seg_train', cv_table=cv_table, wl=wl, limit=trlimit)
    tsts = load_imgs(path + '/seg_test', cv_table=cv_table, wl=wl, limit=tslimit)
    return trns, tsts
    
def in3(path, limit=None):
    cv_table = {
        'buildings': 0,
        'street': 1
    }
    
    wl = ['buildings', 'street']
    
    trns = load_imgs(path + '/seg_train', cv_table=cv_table, wl=wl, limit=trlimit)
    tsts = load_imgs(path + '/seg_test', cv_table=cv_table, wl=wl, limit=tslimit)
    return trns, tsts

def in4(path, limit=None):
    cv_table = {
        'buildings': 0,
        'street': 1,
        'forest': 2,
        'mountain': 3
    }
    
    wl = ['buildings', 'street', 'forest', 'mountain']
    
    trns = load_imgs(path + '/seg_train', cv_table=cv_table, wl=wl, limit=trlimit)
    tsts = load_imgs(path + '/seg_test', cv_table=cv_table, wl=wl, limit=tslimit)
    return trns, tsts

def initialize_dataset(path, limit=None):
    pass
    
    
def nominal_load(path, cv_table=None, wl=None, limit=None):
    cl = next(walk(path))[1]
    # print(next(walk(path + "/")))
    
    if wl is not None:
        cl = list(filter(lambda x: x in wl, cl))
    # print("CL", cl, path)
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

def load_imgs(path, cv_table=None, wl=None, limit=None):
    fls = nominal_load(path, cv_table=cv_table, wl=wl, limit=limit)
    
    return [(load_img(i), j) for i, j in fls]
    

def load_img(path):
    return map_arr(np.array(Image.open(path)).tolist(), lambda x: rgb_gray(*x))
