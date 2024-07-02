import numpy as np
from array_tools import *
from functools import reduce

class InvalidLayerInputException(Exception):
    pass
    
class InvalidLayerOutputException(Exception):
    pass

class InvalidNNSequenceException(Exception):
    pass

class NeuralNetwork:
    def __init__(self, layers):
        if len(layers) > 1:
            for i in range(len(layers) - 1):
                if layers[i].output_m != layers[ i + 1].input_n:
                    raise InvalidNNSequenceException("Layer {0: n} has output length {2: n} and layer {1:n} has input length {3:n}", i, i + 1, layers[i].output_m, layers[i + 1].input_n)
        self.layers = layers
        self.original_layers = layers
    
    def reset(self):
        self.layers = self.original_layers
    
    def train(self, training_data):
        pass
    
    def forward(self, input_val):
        return fw_ll(self.layers, input_val)
    
    def derivative(self, input_val):
        pass
        

class CNN(NeuralNetwork):
    def __init__(self, layers):
        if len(layers) > 1:
            for i in range(len(layers) - 1):
                if layers[i].output_dim != layers[i + 1].input_dim:
                    raise InvalidNNSequenceException("Layer {0: n} has output dimension {2: n} and layer {1:n} has input length {3:n}", i, i + 1, layers[i].output_dim, layers[i + 1].input_dim)
        
        super().__init__(layers)
    
    def train(self, training_data, gamma, iterations=1000):
        
        for it_n in range(iterations):
            for i, j in training_data:
                updateable_layers = list(filter(lambda x: not isinstance(x, StaticLayer), self.layers))
                weight_coords = list(map(lambda x: len(flatten_arr(x.weight)), updateable_layers))
                
                derv = self.get_derivatives(i, j)
                
                for k in range(len(weight_coords)):
                    prevc = sum(weight_coords[:k])
                    e = l_to_array([1 if i >= prevc and i < prevc + weight_coords[k] else 0 for i in range(sum(weight_coords))] + [0]*len(i))

                    rm = mask_arr(derv, e, gdim(updateable_layers[k].weight))

                    nw = unflatten_arr(flatten_arr(rm), gdim(updateable_layers[k].weight))

                    fw = arr_add(updateable_layers[k].weight, arr_mul_sc(nw, -gamma))

                    updateable_layers[k].weight = fw
                    
                    
    def get_derivatives(self, input_val, target_val):
        derl = []
        cval = input_val
        
        
        for ind in range(len(self.layers)):
            if ind < len(self.layers) - 1:
                cl = self.layers[ind]
                rd = cl.derivative(cval)
                cval = cl.forward(cval)
                ncl = self.layers[ind+1]
                if not isinstance(ncl, StaticLayer):
                
                
                    wn, wm = gdim(ncl.weight)
                
                    rd = dglue(id_arr(wn*wm), rd)
                    
                    for pv in range(len(derl)):
                        derl[pv] = dglue(id_arr(wn*wm), derl[pv])
                derl.append(rd)
            else:
                cl = self.layers[ind]
                rd = cl.derivative(cval)
                cval = cl.forward(cval)
                derl.append(rd)
                
        cost_grad = transpose_arr(l_to_array(normalize(l_sub(cval, target_val))))
        
        total_l = [cost_grad] + list(reversed(derl))

        rv = matrix_muls(*total_l)
        
        return rv

class Layer:
    def __init__(self, input_n, output_m, starting_weights):
        self.input_n = input_n
        self.output_m = output_m
        self.weight = starting_weights
    
    def forward(self, input_val):
        if len(input_val) != self.input_n:
            raise InvalidLayerInputException("Received input of length {0:n}, expected {1:n}".format(len(input_val), self.input_n))
        
        result = self.post_forward(self.apply_forward(self.pre_forward(input_val)))

        if len(result) != self.output_m:
            raise InvalidLayerOutputException("Produced output of length {0:n}, expected {1:n}".format(len(result), self.output_m))
            
        return result
    
    def pre_forward(self, input_val):
        pass
    
    def post_forward(self, output_val):
        pass
    
    def apply_forward(self, input_val):
        pass


class CNNLayer(Layer):
    def __init__(self, input_dim, output_dim, starting_weights):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super().__init__(input_dim[0]*input_dim[1], output_dim[0]*output_dim[1], starting_weights)
    
    def pre_forward(self, input_val):
        return unflatten_arr(input_val, self.input_dim)

    def post_forward(self, output_val):
        return flatten_arr(output_val)
    
class StaticLayer(CNNLayer):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim, None)

        
class PerceptronLayer(CNNLayer):
    def __init__(self, input_dim, output_dim, starting_weights):
        super().__init__(input_dim, output_dim, starting_weights)

    def apply_forward(self, input_val):
        r = matrix_muls(self.weight, flatten_arr(input_val))

        return r
    
    def derivative(self, input_val):
        ival = input_val
        n, m = gdim(self.weight)
        
        return hglue(kronecker_prod(transpose_arr(l_to_array(ival)), id_arr(n)), self.weight)
    

class ConvolutionLayer(CNNLayer):
    def __init__(self, input_dim, cv_matrix):
        r = len(cv_matrix)//2
        self.cvc = r
        super().__init__(input_dim, (input_dim[0] - 2*r, input_dim[1] - 2*r), cv_matrix)
        
    def apply_forward(self, input_val):
        return apply_convolution(input_val, self.weight, reduce_dim=True)
    
    def derivative(self, input_val):
        imatrix = unflatten_arr(input_val, self.input_dim)
        n, m = self.input_dim
        
        w_inds = get_windows_reduced_inds(imatrix, gdim(self.weight), (self.cvc, self.cvc))
        l0 = []
        
        for w in flatten_arr(w_inds):
            na, nb = w[0]
            ma, mb = w[1]
            l0.append(flatten_arr(
                [[imatrix[i][j] for j in range(ma, mb)] for i in range(na, nb)]
            ))
            
        l1 = []
        
        for w in flatten_arr(w_inds):
            na, nb = w[0]
            ma, mb = w[1]
            l1.append(flatten_arr(
                [[self.weight[i - na][j - ma] if i >= na and i < nb and j >= ma and j < mb \
                else 0 for j in range(m)] for i in range(n)]
            ))

        return hglue(l0, l1)

class ReLuLayer(StaticLayer):
    def __init__(self, input_dim):
        super().__init__(input_dim, input_dim)

    def apply_forward(self, input_val):
        return map_arr(input_val, flex_relu)
    
    def derivative(self, input_val):
        n = self.input_n
        
        return [[(1 if flex_g0(input_val[i]) else 0) if i == j else 0 for j in range(n)] for i in range(n)]

class SgnLayer(StaticLayer):
    def apply_forward(self, input_val):
        return map_arr(input_val, sgn)

class PoolingLayer(StaticLayer):
    def __init__(self, input_dim, output_dim, operation, opdim, opcenter=(0, 0), def_val=[0, 0, 0]):
        super().__init__(input_dim, output_dim)
        self.operation = operation
        self.opdim = opdim
        self.opcenter = opcenter
        self.def_val = def_val
    
    def apply_forward(self, input_val):
        return apply_operation(input_val, self.opdim, self.opcenter, self.operation, reduce_dim=True, edge_v=self.def_val)

class MaxPoolingLayer(PoolingLayer):
    def __init__(self, input_dim, def_val=[0, 0, 0]):
        output_dim = (input_dim[0] - 1, input_dim[1]-1)
        super().__init__(input_dim, output_dim, max_arr, (2, 2), def_val=def_val)
    
    def derivative(self, input_val):
        imatrix = unflatten_arr(input_val, self.input_dim)
        n, m = self.input_dim
        w_inds = get_windows_reduced_inds(imatrix, (2, 2), (0, 0))
        l = []
        
        for w in flatten_arr(w_inds):
            na, nb = w[0]
            ma, mb = w[1]
            l.append(flatten_arr(
                [[1 if i >= na and i < nb and j >= ma and j < mb and\
                geq_flex(imatrix[i][j], maxv(get_slice_arr(imatrix, (na, nb), (ma, mb))))\
                else 0 for j in range(m)] for i in range(n)]
            ))
        return l


def fw_ll(layer_l, input_val):
    return reduce(lambda v, l: l.forward(v), [input_val] + layer_l)


def flex_g0(x):
    if type(x) is list:
        return all([i > 0 for i in x])
    else:
        return x > 0

def flex_relu(x):
    if type(x) is list:
        return [relu(i) for i in x]
    else:
        return relu(x)

def relu(x):
    return max(0, x)
    
def sgn(x):
    return 1 if x > 0 else -1
    

def stepf(x):
    return (sgn(x)+1)/2
    
    