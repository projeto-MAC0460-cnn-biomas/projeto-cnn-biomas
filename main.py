import numpy as np
from PIL import Image
from array_tools import *
from scipy.ndimage import convolve
from scipy.signal import correlate
import matplotlib.pyplot as plt
from learning import *
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


# barr = gaussian_blur_arr(93)
# imgf = Image.open("ime.jpg")
# imga = np.array(imgf).tolist()
# r = apply_convolution(imga, blur_arr, reduce_dim=False, edge_v=[255, 255, 255])
# ra = np.array([[[0 if k < 0 else k for k in j] for j in i] for i in r], np.uint8)
# ri = Image.fromarray(ra, mode='RGB')
# ri.show()

tfl = PerceptronLayer((3,3), (3, 3), [[1, 1, -1, 1, -1, 1, -1, -1, 1] for i in range(3)] + [[2, 2, 0, 1, 1, -1, 9, 0, -1] for i in range(6)], activation=relu)
pl = MaxPoolingLayer((3, 3))
print(np.array(matrix))
nn = CNN([tfl, pl])
r = nn.forward(flatten_arr(matrix))
print(np.array(r))