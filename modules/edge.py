#!/usr/bin/env python
import abc
import torch


class Edge(abc.ABC):
    def __init__(self,pre_call,pos_call, prenode=None, posnode=None):
        self._nodes = {'pre':prenode,'pos':posnode}
        self.pre_call = (lambda x: x) if pre_call is None else pre_call
        self.pos_call = (lambda x: x) if pos_call is None else pos_call


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
    # TODO: Delete overloading module method workaround to aviod recursive
    # calling.  <15-04-22, Yang Bangcheng> #
    # def parameters(self,*args,**kwargs):
    #     return super().parameters(recurse=False)

    # def named_parameters(self, prefix: str = '', recurse: bool = True):
    #     return super().named_parameters(prefix,recurse=False)

    # def named_modules(self, memo= None, prefix: str = '', remove_duplicate: bool = True):
    #     return super().named_modules(memo={self.pre,self.pos},prefix=prefix,remove_duplicate=remove_duplicate)

    # def _named_members(self, get_members_fn, prefix='', recurse=True):
    #     return super()._named_members(get_members_fn=get_members_fn,prefix=prefix,recurse=False)

    def connect(self, prenode=None, posnode=None):
        self.pre = prenode
        self.pos = posnode

    def init_param_buffer(self):
        for param in self.parameters():
            param.grad_buffer = []

    def __call__(self,input=None):
        if input == None:
            return self.pos_call(self.forward(self.pre_call(self.pre())))
        else:
            return self.pos_call(self.forward(self.pre_call(input)))

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


class Linear(Edge,torch.nn.Linear, ):

    """Docstring for Linear. """

    def __init__(self, in_features, out_features,
                 prenode=None, posnode=None,pre_call=None,pos_call=None, **kwargs):
        """TODO: to be defined. """
        torch.nn.Linear.__init__(
            self,
            in_features,
            out_features,
            **kwargs)
        Edge.__init__(self,pre_call, pos_call, prenode=prenode, posnode=prenode)
        self.init_param_buffer()


class Conv2d( Edge,torch.nn.Conv2d,):

    """Docstring for Conv2d. """

    def __init__(self, in_channels, out_channels, kernel_size,
                 prenode=None, posnode=None,pre_call=None,pos_call=None,**kwargs):
        """TODO: to be defined. """
        torch.nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            **kwargs)
        self.init_param_buffer()
        Edge.__init__(self,pre_call,pos_call, prenode=prenode, posnode=posnode)
