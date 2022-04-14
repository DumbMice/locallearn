#!/usr/bin/env python

import pytest
from modules.node import *
from modules.edge import *
from .fixtures import *


def test_unintialized_state_dtype(NoneNode):
    assert NoneNode.state.dtype == torch.float32

def test_unintialized_state_device(NoneNode):
    assert NoneNode.state.device == torch.device('cpu')

def test_unintialized_state_cuda_device(CudaNoneNode):
    assert CudaNoneNode.state.device == torch.ones(1).cuda().device

def test_double_unintialized_state_dtype(DoubleNode):
    assert DoubleNode.state.dtype == torch.float64

def test_initialized_dim(OnesNode):
    assert OnesNode.dim == torch.Size([3,28,28])

def test_state_set_NoneNode(NoneNode):
    NoneNode.state = torch.ones(10,10)
    assert torch.sum((NoneNode.state.data-torch.ones(10,10))**2,(0,1))==0

def test_state_FilledCudaNode(FilledNode,Ones):
    assert torch.sum((FilledNode.state-Ones)**2)==0

def test_state_FilledNode(FilledCudaNode,CudaOnes):
    assert torch.sum((FilledCudaNode.state-CudaOnes)**2)==0

def test_fill_state_with_common_tensor_grad(NoneNode):
    node1 = Node()
    node2 = Node()
    state = torch.rand(10,10)
    node1.state = state
    node2.state = state
    torch.sum(node1.state**2).backward()
    node1.state=node1.state-node1.state.grad
    assert (node1.state != node2.state).all()

def test_torchsize():
    assert torch.Size([1])+torch.Size([1,1]) == torch.Size([1,1,1])

def test_init_data_NoneNode(NoneNode):
    NoneNode.dim=torch.Size([28,28])
    NoneNode.data_init = torch.nn.init.constant_
    NoneNode.init_data(10,3)
    assert (NoneNode.state.data == torch.ones(10,28,28)*3).all()

def test_init_data_CudaNoneNode(CudaNoneNode):
    CudaNoneNode.dim=torch.Size([28,28])
    CudaNoneNode.data_init = torch.nn.init.constant_
    CudaNoneNode.init_data(10,3)
    assert (CudaNoneNode.state.data == torch.ones(10,28,28).cuda()*3).all()

def test_reset_NoneNode(NoneNode):
    NoneNode.dim=torch.Size([28,28])
    NoneNode.data_init = torch.nn.init.constant_
    NoneNode.init_data(10,3)
    NoneNode.reset(100,2)
    assert (NoneNode.state.data == torch.ones(100,28,28)*2).all()

def test_square_grad_free_nodes(RandomNode):
    result = torch.sum(RandomNode.state**2)
    result.backward()
    assert (RandomNode.state.grad == 2*RandomNode.state.data).all()

def test_exp_grad_free_nodes(RandomNode):
    result = torch.sum(torch.exp(2*RandomNode.state))
    result.backward()
    assert (RandomNode.state.grad == 2*torch.exp(2*RandomNode.state.data)).all()

def test_product_grad_free_nodes(RandomNode):
    tensor = torch.rand(RandomNode().shape,requires_grad=True)
    result = torch.sum(RandomNode()*tensor)
    result.backward()
    assert (RandomNode.state.grad == tensor).all() and (tensor.grad == RandomNode.state.data).all()

def test_product_twice_grad_free_nodes(RandomNode):
    tensor = torch.rand(RandomNode().shape,requires_grad=True)
    result = torch.sum(RandomNode()*tensor)
    result.backward(retain_graph=True)
    result.backward(retain_graph=True)
    assert (RandomNode.state.grad == 2*tensor).all() and (tensor.grad == 2*RandomNode.state.data).all()

def test_product_grad_free_cuda_nodes(RandomCudaNode):
    tensor = torch.rand(RandomCudaNode().shape).cuda()
    tensor.requires_grad_()
    result = torch.sum(RandomCudaNode()*tensor)
    result.backward()
    assert (RandomCudaNode.state.grad == tensor).all() and (tensor.grad == RandomCudaNode.state.data).all()

@pytest.mark.xfail
def test_square_grad_clamped_nodes(RandomNode):
    RandomNode.clamp()
    result = torch.sum(RandomNode.state**2)
    result.backward()
    assert RandomNode.state.grad == 0

def test_product_grad_clamped_nodes(RandomNode):
    tensor = torch.rand(RandomNode().shape,requires_grad=True)
    result = torch.sum(RandomNode()*tensor)
    result.backward(retain_graph=True)
    RandomNode.clamp()
    result.backward(retain_graph=True)
    assert (RandomNode.state.grad == tensor).all() and (tensor.grad == 2*RandomNode.state.data).all()

def test_product_grad_unclamped_nodes(RandomNode):
    tensor = torch.rand(RandomNode().shape,requires_grad=True)
    result = torch.sum(RandomNode()*tensor)
    result.backward(retain_graph=True)
    # gradient accumulates on node

    RandomNode.clamp()
    result.backward(retain_graph=True)
    # gradient stops accumulating on node

    RandomNode.unclamp()
    result.backward(retain_graph=True)
    # gradient accumulates on node

    assert (RandomNode.state.grad == 2*tensor).all() and (tensor.grad == 3*RandomNode.state.data).all()

def test_grad_eye_linear_nodes(EyeLinearEdge,RandomVecNode):
    EyeLinearEdge.pre = RandomVecNode
    result = torch.sum(EyeLinearEdge())
    result.backward()
    assert (RandomVecNode.state.grad==1).all()
