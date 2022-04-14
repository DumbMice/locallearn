#!/usr/bin/env python

import pytest
import copy
from modules.node import *
from modules.edge import *
from modules.network import *
from .fixtures import *


def test_add_node_EP(OnesNode):
    net = EP()
    net.addnode(OnesNode)
    assert net.nodes[0] == OnesNode


def test_add_edge_EP(EyeLinearCudaEdge):
    net = EP()
    net.addedge(EyeLinearCudaEdge)
    assert net.edges[0] == EyeLinearCudaEdge


def test_connect_node_node_EP(NoneNode, OnesNode, EyeLinearCudaEdge):
    net = EP()
    net.connect(NoneNode, OnesNode, EyeLinearCudaEdge)
    assert net.nodes[0] == NoneNode and net.nodes[1] == OnesNode and net.edges[0] == EyeLinearCudaEdge and NoneNode.connectout[
        0] == EyeLinearCudaEdge and OnesNode.connectin[0] == EyeLinearCudaEdge and EyeLinearCudaEdge.pre == NoneNode and EyeLinearCudaEdge.pos == OnesNode


def test_connect_id_id_EP(NoneNode, OnesNode, EyeLinearCudaEdge):
    net = EP()
    net.addnode(NoneNode)
    net.addnode(OnesNode)
    net.connect(0, 1, EyeLinearCudaEdge)
    assert net.nodes[0] == NoneNode and net.nodes[1] == OnesNode and net.edges[0] == EyeLinearCudaEdge and NoneNode.connectout[
        0] == EyeLinearCudaEdge and OnesNode.connectin[0] == EyeLinearCudaEdge and EyeLinearCudaEdge.pre == NoneNode and EyeLinearCudaEdge.pos == OnesNode


def test_set_innode_EP(OnesNode):
    net = EP()
    net.innode = OnesNode
    assert net.innode.clamped == True and net.nodes[0] == OnesNode


def test_set_outnode_EP(OnesNode):
    net = EP()
    net.outnode = OnesNode
    assert net.outnode.clamped == True and len(net.nodes) == 0


def test_cost_MSE_EP(OnesNode):
    net = EP()
    net.costfunc = torch.nn.MSELoss()
    node2 = copy.deepcopy(OnesNode)
    net.addnode(OnesNode)
    net.outnode = node2
    assert net.cost() == 0


def test_addlayerednodes_EP():
    net = EP()
    net.addlayerednodes(10, outnode=False, state=torch.ones(10, 10))
    assert len(net.nodes) == 10 and (net.energy() == 100.).all()


def test_node_step_EP():
    net = EP()
    state = torch.rand(10, 10)
    net.addlayerednodes(3, True, state=state)
    net.node_optim = torch.optim.SGD(net.nodes.parameters(), lr=1)
    net.nodes_step(torch.sum(net.energy()))
    # net.nodes[0] does not require grad, test net.nodes[1]
    assert torch.sum((net.nodes[1].state+state)**2) == 0.

def test_initnodes_EP():
    net = EP()
    net.addlayerednodes(10, True, dim=torch.Size([10]))
    net.initnodes(batch=10)
    assert net.nodes[0].shape == torch.Size([10,10]) and net.nodes[1].shape == torch.Size([10,10])

def test_node_relax_EP(RandomCudaNode):
    net = EP()
    net.addlayerednodes(10, True, dim=torch.Size([10]))
    net.initnodes(batch=10)
    net.node_optim = torch.optim.SGD(net.nodes.parameters(), lr=0.2)
    net.unclamp(0)
    net.etol=1e-8
    etol=net.node_relax(lambda : torch.sum(net.energy()),max_iter=10000)
    assert sum(L2diff(node.state,0) for node in net.nodes)<etol
