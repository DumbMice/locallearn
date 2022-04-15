#!/usr/bin/env python

import pytest
from modules.node import *
from modules.edge import *
from modules.network import *

@pytest.fixture
def NoneNode():
    return Node()

@pytest.fixture
def CudaNoneNode():
    return Node().cuda()

@pytest.fixture
def DoubleNode():
    return Node(dtype=torch.float64)

@pytest.fixture
def CudaDoubleNode():
    return Node(dtype=torch.float64).cuda()

@pytest.fixture
def OnesNode():
    return Node(state=torch.ones(10,3,28,28))

@pytest.fixture
def FilledNode():
    node=Node(dtype=torch.float64)
    node.state = torch.ones(10,3,28,28)
    return node

@pytest.fixture
def FilledCudaNode():
    node=Node(dtype=torch.float64).cuda()
    node.state = torch.ones(10,3,28,28).cuda()
    return node

@pytest.fixture
def Ones():
    return torch.ones(10,3,28,28)

@pytest.fixture
def CudaOnes():
    return torch.ones(10,3,28,28).cuda()

@pytest.fixture
def RandomNode():
    return Node(state=torch.rand(10,3,28,28))

@pytest.fixture
def RandomCudaNode():
    return Node(state=torch.rand(10,3,28,28)).cuda()

@pytest.fixture
def VecNode():
    return Node(state=torch.ones(10,10))

@pytest.fixture
def VecCudaNode():
    return Node(state=torch.ones(10,10)).cuda()

@pytest.fixture
def RandomVecNode():
    return Node(state=torch.rand(10,10))

@pytest.fixture
def RandomVecCudaNode():
    return Node(state=torch.rand(10,10)).cuda()

@pytest.fixture
def RandomLinearEdge():
    return Linear(10,20)

@pytest.fixture
def RandomLinearCudaEdge():
    return Linear(10,20).cuda()

@pytest.fixture
def EyeLinearEdge():
    linear=Linear(10,10,bias=False)
    torch.nn.init.eye_(linear.weight)
    return linear

@pytest.fixture
def EyeLinearCudaEdge():
    linear=Linear(10,10,bias=False).cuda()
    torch.nn.init.eye_(linear.weight)
    return linear

@pytest.fixture
def EP10():
    net = EP()
    net.addlayerednodes(10,True)
    net.connect(0,1,Linear(10,10))
    net.connect(1,2,Linear(10,10))
    net.connect(2,3,Linear(10,10))
    net.connect(3,4,Linear(10,10))
    net.connect(4,5,Linear(10,10))
    net.connect(5,6,Linear(10,10))
    net.connect(6,7,Linear(10,10))
    net.connect(7,8,Linear(10,10))
    net.connect(8,9,Linear(10,10))
    return net

@pytest.fixture
def CudaEP10():
    net = EP()
    net.addlayerednodes(10,True)
    net.connect(0,1,Linear(10,10))
    net.connect(1,2,Linear(10,10))
    net.connect(2,3,Linear(10,10))
    net.connect(3,4,Linear(10,10))
    net.connect(4,5,Linear(10,10))
    net.connect(5,6,Linear(10,10))
    net.connect(6,7,Linear(10,10))
    net.connect(7,8,Linear(10,10))
    net.connect(8,9,Linear(10,10))
    net.to(torch.device("cuda"))
    return net

@pytest.fixture
def CudaEP10Conv():
    device = torch.ones(1).cuda().device
    net = EP()
    net.addlayerednodes(10,True)
    net.connect(0,1,Conv2d(3,10,3))
    net.connect(1,2,Conv2d(10,10,3))
    net.connect(2,3,Conv2d(10,10,3))
    net.connect(3,4,Conv2d(10,11,3))
    net.connect(4,5,Conv2d(11,12,3))
    net.connect(5,6,Conv2d(12,13,3))
    net.connect(6,7,Conv2d(13,13,3))
    net.connect(7,8,Conv2d(13,21,3))
    net.connect(8,9,Conv2d(21,22,3))
    net.to(device)
    return net

def L2diff(tensor1,tensor2):
    return torch.sum((tensor1-tensor2)**2)

def AllEqual(tensor1,tensor2):
    return (tensor1 == tensor2).all()

