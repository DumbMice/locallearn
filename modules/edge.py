#!/usr/bin/env python
import abc
import torch

class Edge(abc.ABC, torch.nn.Module):
    def __init__(self, prenode=None,posnode=None):
        self.pre = prenode
        self.pos = posnode
        super().__init__()

    def connect(self,prenode=None,posnode=None):
        self.pre = prenode
        self.pos = posnode

    def init_param_buffer(self):
        for param in self.parameters(recurse=False):
            param.grad_buffer = []

    def __call__(self):
        return self.forward(self.pre())

    def storegrad(self):
        for param in self.parameters(recurse=False):
            param.grad_buffer.append(param.grad)

    def addgrad(self,index):
        for param in self.parameters(recurse=False):
            param.grad += param.grad_buffer[index]

    def popgrad(self,index):
        for param in self.parameters(recurse=False):
            param.grad_buffer.pop(index)

    def freeze_grad(self):
        for param in self.parameters(recurse=False):
            param.requires_grad_(False)

    def free_grad(self):
        for param in self.parameters(recurse=False):
            param.requires_grad_(True)

class Linear(torch.nn.Linear,Edge,):

    """Docstring for Linear. """

    def __init__(self,in_features,out_features,prenode=None,posnode=None,*args,**kwargs):
        """TODO: to be defined. """
        Edge.__init__(self,prenode,posnode)
        torch.nn.Linear.__init__(self,in_features,out_features,*args,**kwargs)
        self.init_param_buffer()

class Conv2d(torch.nn.Conv2d,Edge):

    """Docstring for Conv2d. """

    def __init__(self,in_channels,out_channels,kernel_size,prenode=None,posnode=None,*args,**kwargs):
        """TODO: to be defined. """
        Edge.__init__(self,prenode,posnode)
        torch.nn.Conv2d.__init__(self,in_channels,out_channels,kernel_size,*args,**kwargs)
        self.init_param_buffer()
