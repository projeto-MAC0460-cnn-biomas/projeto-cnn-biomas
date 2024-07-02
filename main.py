import numpy as np
from PIL import Image
from array_tools import *
from scipy.ndimage import convolve
from scipy.signal import correlate
import matplotlib.pyplot as plt
from learning import *
from array_tools import *
from dataset import *
from copy import deepcopy


def proc_bansw(a):
    return round(min(0, max(1, a)))

def gen_perceptron_layers(dim, n):
    l = []
    
    for i in range(n):
        l.append(PerceptronLayer(dim, dim, id_arr(dim[0]*dim[1])))
    
    return l

layer_l = [
ConvolutionLayer((150, 150), one_middle_arr(27)),
ReLuLayer((124, 124)),
MaxPoolingLayer((124, 124)),
ConvolutionLayer((123, 123), one_middle_arr(25)),
ReLuLayer((99, 99)),
MaxPoolingLayer((99, 99)),
ConvolutionLayer((98, 98), one_middle_arr(25)),
ReLuLayer((74, 74)),
ConvolutionLayer((74, 74), one_middle_arr(21)),
ReLuLayer((54, 54)),
ConvolutionLayer((54, 54), one_middle_arr(21)),
ReLuLayer((34, 34)),
MaxPoolingLayer((34, 34))
] + gen_perceptron_layers((33, 33), 50) + [PerceptronLayer((33, 33), (1, 1), [[1 for i in range(34*34)]]), StepLayer((1, 1))]

# Treina os três modelos

c1 = CNN(layer_l)
c2 = deepcopy(c1)
c3 = deepcopy(c1)

tr_ds1,ts_ds1 = in1(pth, trlimit = 1500,tslimit=600)

c1.train(tr_ds1, gamma=0.0.01, iterations=10000)

tr_ds2,ts_ds2 = in2(pth, trlimit = 1500,tslimit=300)

c2.train(tr_ds2, gamma=0.0.1, iterations=10000)

tr_ds3,ts_ds3 = in1(pth, trlimit = 1500,tslimit=300)

c3.train(tr_ds3, gamma=0.0.1, iterations=10000)

# c4
def classify(input_val):
    r1 = c1.forward(input_val)
    
    # natureza
    if r1 == 0:
        r2 = c2.forward(input_val)
    # concreto
    else:
        r2 = c3.forward(input_val)
        
    if r1 == 0 and r2 == 0:
        return cv_t['forest']
    elif r1 == 0 and r2 == 1:
        return cv_t['mountain']
    elif r1 == 1 and r2 == 0:
        return cv_t['buildings']
    else:
        return cv_t['street']



def t_accuracy(fw_f, test_set):
    c = 0
    
    for i, j in test_set:
        k = fw_f(i)
        
        if j == k:
            c += 1
            
    return (c, len(test_set), c/len(test_set))

# Testa os modelos

print(t_accuracy(c1.forward, ts_ds1))
print(t_accuracy(c2.forward, ts_ds2))
print(t_accuracy(c3.forward, ts_ds3))
print(t_accuracy(classify, ts_ds4))