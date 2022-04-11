#!/usr/bin/env python

import pytest
from locallearn import *

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
