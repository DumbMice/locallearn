#!/usr/bin/env python
import abc
import torch
from . import function

class Edge(abc.ABC):
    def __init__(self,*args,**kwargs):
        self._nodes = {'pre':None,'pos':None}
        super().__init__(*args,**kwargs)
        self.init_param_buffer()

    def __setattr__(self, name, value):
        """
        Overwrite __setattr__ to bypass default setattr behavior of torch.nn.Module
        on innode and outnode
        """
        if name == 'pre':
            self._nodes['pre']=value
        elif name == 'pos':
            self._nodes['pos']=value
        else:
            super().__setattr__(name, value)

    @property
    def pre(self):
        return self._nodes['pre']

    @property
    def pos(self):
        return self._nodes['pos']

    def energy(self):
        return None

    def connect(self, prenode=None, posnode=None):
        self.pre = prenode
        self.pos = posnode

    def init_param_buffer(self):
        for param in self.parameters():
            param.grad_buffer = []

    def __call__(self,input=None,reverse=False):
        if input == None:
            return self.forward(self.pre())
        else:
            return self.forward(input)

    def store_grad(self):
        for param in self.parameters():
            param.grad_buffer.append(param.grad.detach().clone())

    def add_grad(self, index):
        for param in self.parameters():
            param.grad += param.grad_buffer[index]

    def pop_grad(self, index):
        for param in self.parameters():
            # TODO: del tensor  <12-04-22, Yang Bangcheng> #
            param.grad_buffer.pop(index)

    def freeze_grad(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def free_grad(self):
        for param in self.parameters():
            param.requires_grad_(True)

def EdgeBuilder(module,*args,**kwargs):
    class EdgeModule(Edge,module):
        def __init__(self, *args,**kwargs):
            """TODO: to be defined. """
            super().__init__(*args,**kwargs)

    return EdgeModule(*args,**kwargs)

class Linear(Edge,function.Linear ):

    """Docstring for Linear. """

    def __init__(self, *args,**kwargs):
        """TODO: to be defined. """
        super().__init__(*args,**kwargs)


class Conv2d( Edge,function.Conv2d,):

    """Docstring for Conv2d. """

    def __init__(self, *args,**kwargs):
        """TODO: to be defined. """
        super().__init__(*args,**kwargs)

class PatchEmbedding( Edge,function.PatchEmbedding):

    """Docstring for Conv2d. """

    def __init__(self, *args,**kwargs):
        """TODO: to be defined. """
        super().__init__(*args,**kwargs)

class Sequential(Edge,function.Sequential):
    def __init__(self, *args,**kwargs):
        """TODO: to be defined. """
        super().__init__(*args,**kwargs)

class Flatten(Edge,function.Flatten):
    def __init__(self, *args,**kwargs):
        """TODO: to be defined. """
        super().__init__(*args,**kwargs)

class MaxPool2d(Edge,function.MaxPool2d):
    def __init__(self, *args,**kwargs):
        """TODO: to be defined. """
        super().__init__(*args,**kwargs)

class ReLU(Edge,function.ReLU):
    def __init__(self, *args,**kwargs):
        """TODO: to be defined. """
        super().__init__(*args,**kwargs)

class LeakyReLU(Edge,function.LeakyReLU):
    def __init__(self, *args,**kwargs):
        """TODO: to be defined. """
        super().__init__(*args,**kwargs)
    # return input+(0.5-0.5*torch.abs(input)/input)*(1./self.negative_slope-1.)*input
class SELU(Edge,function.SELU):
    def __init__(self, *args,**kwargs):
        """TODO: to be defined. """
        super().__init__(*args,**kwargs)

class Sigmoid(Edge,function.Sigmoid):
    def __init__(self, *args,**kwargs):
        """TODO: to be defined. """
        super().__init__(*args,**kwargs)

class Tanh(Edge,function.Tanh):
    def __init__(self, *args,**kwargs):
        """TODO: to be defined. """
        super().__init__(*args,**kwargs)

class BatchNorm2d(Edge,function.BatchNorm2d):
    def __init__(self, *args,**kwargs):
        """TODO: to be defined. """
        super().__init__(*args,**kwargs)
