import numpy as np
from PIL import Image
from math import exp
from functools import reduce
from math import sqrt

def get_windows(arr, dim, center_pos, reduce_dim=True, edge_v=0):
    if reduce_dim:
        return get_windows_reduced(arr, dim, center_pos)
    else:
        return get_windows_reduced(
            pad_arr(arr,
                        (center_pos[0],
                        dim[0] - (center_pos[0] + 1)),
                        (center_pos[1],
                        dim[1] - (center_pos[1] + 1)),
                        edge_v),
            dim, center_pos)
    

def get_windows_reduced(arr, dim, center_pos):
    matrix_n, matrix_m = gdim(arr)
    start_n, start_m = center_pos
    end_n = matrix_n - (dim[0] - (center_pos[0] + 1))
    end_m = matrix_m - (dim[1] - (center_pos[1] + 1))
    return [[
        get_slice_arr(arr, 
        (i - center_pos[0], i+(dim[0] - (center_pos[0] + 1) + 1)),
        (j - center_pos[1], j+(dim[1] - (center_pos[1] + 1) + 1)))
             for j in range(start_m, end_m)] for i in range(start_n, end_n)]

def get_windows_reduced_inds(arr, dim, center_pos):
    matrix_n, matrix_m = gdim(arr)
    start_n, start_m = center_pos
    end_n = matrix_n - (dim[0] - (center_pos[0] + 1))
    end_m = matrix_m - (dim[1] - (center_pos[1] + 1))
    return [[( 
        (i - center_pos[0], i+(dim[0] - (center_pos[0] + 1) + 1)),
        (j - center_pos[1], j+(dim[1] - (center_pos[1] + 1) + 1))
        ) for j in range(start_m, end_m)] for i in range(start_n, end_n)]
 
def apply_operation(arr, dim, center_pos, op, reduce_dim=True, edge_v=0):
    rm = get_windows(arr, dim, center_pos, reduce_dim=reduce_dim, edge_v=edge_v)
    n, m = gdim(rm)
    
    return [[op(rm[i][j]) for j in range(m)] for i in range(n)]

def apply_convolution(arr, convol_arr, reduce_dim=False, edge_v=0):
    f = lambda x: sum_arr(arr_mul(x, convol_arr))
    midp = len(convol_arr) // 2
    return apply_operation(arr, gdim(convol_arr), (midp, midp), f, reduce_dim=reduce_dim, edge_v=edge_v)

def get_slice_arr(arr, ni, mi):
    return [j[mi[0]:mi[1]] for j in arr[ni[0]:ni[1]]]

def glen(l):
    return len(l) if type(l) is list else 1

def gdim(arr):
    return (glen(arr), glen(arr[0]))


def flatten_arr(arr):

    n, m = gdim(arr)
    return [arr[i][j] for j in range(m) for i in range(n)]

def unflatten_arr(l, dim):
    return [l[i*dim[1]:(i+1)*dim[1]] for i in range(len(l)//dim[1])]

def pad_arr(arr, n_pad, m_pad, val):
    n, m = gdim(arr)
    
    return [
    [
        (val if (i < n_pad[0] or j < m_pad[0] or i >= n_pad[0] + n or j >= m_pad[0] + m) else arr[i - n_pad[0]][j - m_pad[0]] ) for j in range(m + m_pad[0] + m_pad[1])
    ] for i in range(n + n_pad[0] + n_pad[1])
    ]

def arr_add_sc(arr, v):
    return [[j+v for j in i] for i in arr]

def arr_mul_sc(arr, v):
    return [[j*v for j in i] for i in arr]

def arr_add(arr1, arr2):
    return [[add_flex(arr1[i][j], arr2[i][j]) for j in range(len(arr1[i]))] for i in range(len(arr1))]
    
def arr_mul(arr1, arr2):
    return [[mul_flex(arr1[i][j],arr2[i][j]) for j in range(len(arr1[i]))] for i in range(len(arr1))]

def l_add(v1, v2):
    return [v1[i] + v2[i] for i in range(len(v1))]

def l_add_sc(v1, v2):
    return [v1[i] + v2 for i in range(len(v1))]

def add_flex(v1, v2):
    if type(v1) is list and type(v2) is list:
        return l_add(v1, v2)
    elif type(v1) is list and type(v2) is not list:
        return l_add_sc(v1, v2)
    elif type(v1) is not list and type(v2) is list:
        return l_add_sc(v2, v1)
    else:
        return v1 + v2

def l_mul(v1, v2):
    return [v1[i] * v2[i] for i in range(len(v1))]

def l_mul_sc(v1, v2):
    return [v1[i] * v2 for i in range(len(v1))]

def mul_flex(v1, v2):
    if type(v1) is list and type(v2) is list:
        return l_mul(v1, v2)
    elif type(v1) is list and type(v2) is not list:
        return l_mul_sc(v1, v2)
    elif type(v1) is not list and type(v2) is list:
        return l_mul_sc(v2, v1)
    else:
        return v1 * v2

def sum_arr(arr):

    return sumv(flatten_arr(arr))

def max_arr(arr):
    return maxv(flatten_arr(arr))

def sumv(l):
    if type(l) is not list:
        return l
    
    r = None
    
    for i in range(len(l)):
        if i == 0:
            r = l[i]
        else:
            r = add_flex(r, l[i])
    return r

def max_flex(v1, v2):
    if type(v1) is list and type(v2) is list:
        return l_max(v1, v2)
    elif type(v1) is list and type(v2) is not list:
        return l_max_sc(v1, v2)
    elif type(v1) is not list and type(v2) is list:
        return l_max_sc(v2, v1)
    else:
        return max(v1, v2)

def l_max(v1, v2):
    return [max(v1[i], v2[i]) for i in range(len(v1))]

def l_max_sc(v1, v2):
    return [max(v1[i], v2) for i in range(len(v1))]

def maxv(l):
    r = None
    
    for i in range(len(l)):
        if i == 0:
            r = l[i]
        else:
            r = max_flex(r, l[i])
    return r

def gaussian_blur_arr(n):
    center = n // 2
    
    return [[exp(-((j-center)**2+(i-center)^2)/n) for j in range(n)] for i in range(n)]

def flen(arr):
    if type(arr) is list:
        if len(arr) == 0:
            return 0
        else:
            return len(arr)*flen(arr[0])
    else:
        return 1

def geq_flex(v1, v2):
    if type(v1) is list and type(v2) is list:
        return all([v1[i] >= v2[i] for i in range(len(v1))])
    elif type(v1) is list and type(v2) is not list:
        return all([v1[i] >= v2 for i in range(len(v1))])
    elif type(v1) is not list and type(v2) is list:
        return all([v1 >= v2[i] for i in range(len(v2))])
    else:
        return v1 >= v2

def transpose_arr(arr):
    n, m = gdim(arr)
    return [[arr[i][j] for i in range(n)] for j in range(m)]

def l_to_array(l):
    n, m = gdim(l)
    
    if m == 1 and type(l[0]) is not list:
        return [[i] for i in l]
    else:
        return l

def matrix_mul(a, b):
    a = l_to_array(a)
    b = l_to_array(b)
    n1, m1 = gdim(a)
    n2, m2 = gdim(b)
    return [[sumv([mul_flex(a[i][k], b[k][j]) for k in range(m1)]) for j in range(m2)] for i in range(n1)]

def matrix_muls(*l):
    return reduce(matrix_mul, l)

def splitl(l, n):
    return (l[:n], l[n:])

def map_arr(arr, f):
    return [[f(j) for j in i] for i in arr]

def kronecker_prod(a, b):
    m, n = gdim(a)
    p, q = gdim(b)
    
    return [[
        a[i//p][j//q]*b[i%p][j%q]
    for j in range(q*n)] for i in range(p*m)]

def hglue(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def vglue(a, b):
    return a + b

def id_arr(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def dist_flex(v1, v2):
    return mdnorm(sub_flex(v1, v2))
    
def mdnorm(l):
    return norm(flatten_arr(l))

def norm(l):
    return sqrt(sum([i**2 for i in l]))

def l_sub(v1, v2):
    return [v1[i] - v2[i] for i in range(len(v1))]

def l_sub_sc_l(v1, v2):
    return [v1 - v2[i] for i in range(len(v1))]

def l_sub_sc_r(v1, v2):
    return [v1[i] - v2 for i in range(len(v1))]

def sub_flex(v1, v2):
    if type(v1) is list and type(v2) is list:
        return l_sub(v1, v2)
    elif type(v1) is list and type(v2) is not list:
        return l_sub_sc_r(v1, v2)
    elif type(v1) is not list and type(v2) is list:
        return l_sub_sc_l(v2, v1)
    else:
        return v1 - v2

def rgb_gray(r, g, b):
    return max(1, 0.226*r + 0.7152*g + 0.0722*b)

def zeroes_arr(n, m):
    return [[0 for j in range(m)] for i in range(n)]
    
def one_middle_arr(n):
    r = [[0 for j in range(n)] for i in range(n)]
    r[n//2][n//2] = 1
    return r

def dglue(a, b):
    n, m = gdim(a)
    p, q = gdim(b)
    
    up_arr = hglue(a, zeroes_arr(n, q))
    down_arr = hglue(zeroes_arr(p, m), b)
    
    return vglue(up_arr, down_arr)

def normalize(v):
    return [i/norm(v) for i in v]


def mask_arr(m, e, dim):
    l = list(zip(flatten_arr(m), flatten_arr(e)))
    rl = list(map(lambda x: x[0], filter(lambda x: x[1] > 0, l)))
    
    return unflatten_arr(rl, dim)