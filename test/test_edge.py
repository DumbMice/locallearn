#!/usr/bin/env python

import pytest
from modules.node import *
from modules.edge import *
from .fixtures import *


def test_linear_initilized_param_shape(RandomLinearEdge):
    assert(RandomLinearEdge.weight.shape == torch.Size([20, 10])) and (
        RandomLinearEdge.bias.shape == torch.Size([20]))

def test_linear_connected_param_shape(RandomLinearEdge,OnesNode,FilledNode):
    RandomLinearEdge.connect(OnesNode,FilledNode)
    assert len(list(RandomLinearEdge.parameters()))==2

def test_linear_connected_param_shape(RandomLinearEdge,OnesNode,FilledNode):
    RandomLinearEdge.connect(OnesNode,FilledNode)
    assert len(list(RandomLinearEdge._parameters.items()))==2

def test_linear_connected_module_number(RandomLinearEdge,OnesNode,FilledNode):
    RandomLinearEdge.connect(OnesNode,FilledNode)
    print(f'named module is {list(RandomLinearEdge.named_modules())} with length {len(list(RandomLinearEdge.named_modules()))}')
    assert len(list(RandomLinearEdge.modules()))==1

def test_eye_linear_forward(EyeLinearEdge, RandomVecNode):
    EyeLinearEdge.pre = RandomVecNode
    assert ((EyeLinearEdge() == RandomVecNode())).all

def test_grad_eye_linear_edges(EyeLinearEdge, RandomVecNode):
    EyeLinearEdge.pre = RandomVecNode
    result = torch.sum(EyeLinearEdge())
    result.backward()
    assert torch.sum((EyeLinearEdge.weight.grad-torch.sum(RandomVecNode(),
                     0).expand(RandomVecNode().shape[0], -1))**2) < 1e-4


def test_grad_eye_linear_cuda_edges(EyeLinearCudaEdge, RandomVecCudaNode):
    EyeLinearCudaEdge.pre = RandomVecCudaNode
    result = torch.sum(EyeLinearCudaEdge())
    result.backward()
    assert torch.sum(
        (EyeLinearCudaEdge.weight.grad-torch.sum(
            RandomVecCudaNode(), 0).expand(
            RandomVecCudaNode().shape[0], -1))**2) < 1e-4


def test_grad_eye_linear_freeze_cuda_edges(
        EyeLinearCudaEdge, RandomVecCudaNode):
    EyeLinearCudaEdge.pre = RandomVecCudaNode
    result = torch.sum(EyeLinearCudaEdge())
    result.backward(retain_graph=True)
    EyeLinearCudaEdge.freeze_grad()
    result.backward(retain_graph=True)
    assert torch.sum(
        (EyeLinearCudaEdge.weight.grad-torch.sum(
            RandomVecCudaNode(), 0).expand(
            RandomVecCudaNode().shape[0], -1))**2) < 1e-4


def test_grad_eye_linear_free_cuda_edges(EyeLinearCudaEdge, RandomVecCudaNode):
    EyeLinearCudaEdge.pre = RandomVecCudaNode
    result = torch.sum(EyeLinearCudaEdge())
    result.backward(retain_graph=True)
    EyeLinearCudaEdge.freeze_grad()
    result.backward(retain_graph=True)
    EyeLinearCudaEdge.free_grad()
    result.backward(retain_graph=True)
    assert torch.sum(
        (EyeLinearCudaEdge.weight.grad-2*torch.sum(
            RandomVecCudaNode(), 0).expand(
            RandomVecCudaNode().shape[0], -1))**2) < 1e-4


def test_grad_buffer_edges(EyeLinearCudaEdge):
    assert EyeLinearCudaEdge.weight.grad_buffer == []


def test_store_grad_edges(EyeLinearCudaEdge, RandomVecCudaNode):
    EyeLinearCudaEdge.pre = RandomVecCudaNode
    result = torch.sum(EyeLinearCudaEdge())
    result.backward()
    params = list(EyeLinearCudaEdge.parameters())
    EyeLinearCudaEdge.store_grad()
    assert (EyeLinearCudaEdge.weight.grad ==
            EyeLinearCudaEdge.weight.grad_buffer[0]).all()

def test_add_grad_edges(EyeLinearCudaEdge, RandomVecCudaNode):
    EyeLinearCudaEdge.pre = RandomVecCudaNode
    result = torch.sum(EyeLinearCudaEdge())
    result.backward()
    params = list(EyeLinearCudaEdge.parameters())
    EyeLinearCudaEdge.store_grad()
    EyeLinearCudaEdge.add_grad(0)
    assert (EyeLinearCudaEdge.weight.grad ==
            2*EyeLinearCudaEdge.weight.grad_buffer[0]).all()


