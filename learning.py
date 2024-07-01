import numpy as np
from array_tools import flen, max_arr, apply_operation, apply_convolution, transpose_arr, matrix_muls, flatten_arr, unflatten_arr, map_arr
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
                new_weights = list(range(len(self.layers)))
                
                for k in range(len(self.layers)):
                    if isinstance(self.layers[k], StaticLayer):
                        continue
                    
                    a, b = splitl(self.layers, k)
                    
                    cw = self.layers[k].weight
                    new_weights[k] = cw - gamma * matrix_muls(
                        transpose_arr(ll_fw(a, input_val)), ll_fw(self.layers, input_val), ll_fw
                    )


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
    def __init__(self, input_dim, output_dim, starting_weights, activation=None):
        if activation is not None:
            self.activation = activation
        else:
            self.activation = None
        super().__init__(input_dim, output_dim, starting_weights)

    def apply_forward(self, input_val):
        r = matrix_muls(self.weight, flatten_arr(input_val))

        if self.activation is not None:
            r = map_arr(r, self.activation)

        return r

class ConvolutionLayer(CNNLayer):
    def __init__(self, input_dim, output_dim, operation):
        super().__init__(input_dim, output_dim, operation)

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



def fw_ll(layer_l, input_val):
    return reduce(lambda v, l: l.forward(v), [input_val] + layer_l)


def relu(x):
    return max(0, x)
    
def sgn(x):
    return 1 if x > 0 else -1
    

def stepf(x):
    return (sgn(x)+1)/2
    