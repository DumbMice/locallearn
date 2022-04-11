#!/usr/bin/env python

import abc
import torch
from functools import singledispatchmethod


class Node(abc.ABC, torch.nn.Module):
    """
    Abstract Base Class for all Node
    Attributes:
        state: pytorch parameters that can be optimized thorugh pytorch optmizer.
        connections: iterable or dict of edges connected to the node frozen: boolean indicating whether the state should be updated
    """

    def __init__(self, state=None, dim=None, device=None, dtype=None,
                 clamped=False, data_init=torch.nn.init.xavier_uniform_):
        super().__init__()
        if state is None:
            self._state = torch.nn.parameter.UninitializedParameter(
                requires_grad=not clamped, dtype=dtype, device=device)
        else:
            self._state = torch.nn.parameter.Parameter(state)
        self.clamped = clamped
        self.connectin = {}
        self.connectout = {}
        self._dim = dim
        self.data_init = data_init
        self._batch = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if len(value.shape) < 2:
            raise Warning(
                f'The number of dimension of state has to be at least 2, but got {len(value.shape)}!')
        if self.dim is not None and self.dim != value.shape[1::]:
            raise Warning(
                f'The state of dim {self.dim} is set by a tensor of dim {value.shape[1::]}')
        if isinstance(
                self.state, torch.nn.parameter.UninitializedParameter):
            self.state.materialize(value.shape)
        self.state.data = value
        self._batch = self.batch

    @property
    def shape(self):
        if not isinstance(
                self.state, torch.nn.parameter.UninitializedParameter):
            return self._state.shape
        return None

    @property
    def dim(self):
        if not isinstance(
                self.state, torch.nn.parameter.UninitializedParameter):
            return self._state.shape[1::]
        return self._dim

    @property
    def batch(self):
        if not isinstance(
                self.state, torch.nn.parameter.UninitializedParameter):
            return self._state.shape[0]
        return self._batch

    @dim.setter
    def dim(self, value):
        if self.dim is None:
            self._dim = value
        elif self.dim != value:
            raise Exception(
                f'dim has already been set to {self.dim}. This new dim {value} is in conflict with the orignal one.')

    def __call__(self):
        return self.state

    def init_data(self, batch=None, *args, **kwargs):
        assert isinstance(
            self.state,
            torch.nn.parameter.UninitializedParameter)
        assert self.dim is not None
        if batch is None:
            batch = self.batch
        self.state.materialize(torch.Size([batch])+self.dim)
        self.data_init(self.state, *args, **kwargs)

    def reset(self, batch, *args, **kwargs):
        if self.clamped:
            return
        assert self.batch is not None
        assert self.dim is not None
        if self.batch != batch:
            self.state = torch.empty(
                torch.Size([batch]) + self.dim, dtype=self.state.dtype,
                device=self.state.device,
                requires_grad=self.state.requires_grad)
        self.data_init(self.state, *args, **kwargs)
        return self.state

    def clamp(self):
        self.state.requires_grad_(False)
        self.clamped = True

    def unclamp(self):
        self.state.requires_grad_(True)
        self.clamped = False

    def checkin(self,edge):
        ##########
        #  TODO  #
        ##########

        # if self.dim is None or (self.dim == edge.requireout())
        #     return True
        # else:
        #     return False
        pass

    def checkout(self,edge):
        ##########
        #  TODO  #
        ##########

        # if self.dim is None or (self.dim == edge.requirein())
        #     return True
        # else:
        #     return False
        pass


