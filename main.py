import numpy as np
from PIL import Image
from array_tools import *
from scipy.ndimage import convolve
from scipy.signal import correlate
import matplotlib.pyplot as plt
from learning import *
from array_tools import *
#imga = plt.imread("ime.jpg")


blur_arr = arr_mul_sc([[1, 4, 6, 4, 1],
                            [4, 16, 24, 16, 4],
                            [6, 24, 36, 24, 6],
                            [4, 16, 24, 16, 4],
                            [1, 4, 6, 4, 1]], 1/256)

ridge_arr = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
matrix = [[-1, 2, 4], [2, 3, 1], [7, 9, 11]]

#r = get_windows(matrix, (2, 2), (1, 1), reduce_dim=False)
cva = [[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]]
zm = [[2, 3, 4, -5, 8], [1, 7, 0, -7, -2], [10, 4, -1, 0, 0], [-1, 0, 6, 7, 4], [4, 2, 2, 3, -1]]

def pm(m):
    print(np.array(m))

def simple_cnn():
    l1 = ConvolutionLayer((5, 5), ridge_arr)
    l2 = MaxPoolingLayer((3, 3))
    l3 = ReLuLayer((2, 2))
    l4 = PerceptronLayer((2, 2), (1, 1), [[1, 1, -1, -1]])
    
    ts = [(flatten_arr(zm), [2])]
    
    netw = CNN([l1, l2, l3, l4])
    
    for i, j in ts:
        r = netw.forward(i)
        print(j, r)
        """
        drv = netw.get_derivatives(flatten_arr(i), j)
        print("Derivative")
        
        pm(drv)
        print(gdim(drv))
        updateable_layers = list(filter(lambda x: not isinstance(x, StaticLayer), netw.layers))
        weight_coords = list(map(lambda x: len(flatten_arr(x.weight)), updateable_layers))
                
        derv = drv
        gamma = 0.01
        for k in range(len(weight_coords)):
            prevc = sum(weight_coords[:k])
            e = l_to_array([1 if i >= prevc and i < prevc + weight_coords[k] else 0 for i in range(sum(weight_coords))] + [0]*len(flatten_arr(i)))
            pm(transpose_arr(e))
            rm = mask_arr(derv, e, gdim(updateable_layers[k].weight))

            nw = unflatten_arr(flatten_arr(rm), gdim(updateable_layers[k].weight))
            pm(nw)
            pm(updateable_layers[k].weight)
            fw = arr_add(updateable_layers[k].weight, arr_mul_sc(nw, -gamma))
            pm(fw)
            updateable_layers[k].weight = fw
    """
    
    netw.train(ts,0.01,iterations=5000)
    
    for i, j in ts:
        r = netw.forward(i)
        print(j, r)
simple_cnn()