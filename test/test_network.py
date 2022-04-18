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


def test_nodes_parameters_EP(EP10):
    node_params = list(EP10.nodes.parameters())
    assert len(EP10.nodes) == len(node_params)


def test_edge_parameters_EP(EP10):
    edge_params = list(EP10.edges.named_parameters())
    # print(edge_params)
    assert 2*len(EP10.edges) == len(edge_params)


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
    assert len(net.nodes) == 10 and (net.energy() == 50.).all()


def test_node_step_EP():
    net = EP()
    state = torch.rand(10, 10)
    net.addlayerednodes(3, True, state=state)
    net.node_optim = torch.optim.SGD(net.nodes.parameters(), lr=1)
    net.nodes_step(torch.sum(net.energy()))
    # net.nodes[0] does not require grad, test net.nodes[1]
    assert torch.sum((net.nodes[1].state)**2) == 0.


def test_initnodes_EP():
    net = EP()
    net.addlayerednodes(10, True, dim=torch.Size([10]))
    net.initnodes(batch=10)
    assert net.nodes[0].shape == torch.Size(
        [10, 10]) and net.nodes[1].shape == torch.Size(
        [10, 10])


def test_node_relax_without_edge_EP():
    net = EP()
    net.addlayerednodes(10, True, dim=torch.Size([10]))
    net.initnodes(batch=10)
    net.node_optim = torch.optim.SGD(net.nodes.parameters(), lr=0.2)
    net.unclamp(0)
    net.etol = 1e-8
    etol, elast = net.node_relax(
        lambda: torch.sum(
            net.energy()), max_iter=10000)
    assert elast > etol and elast < 1e-5


def test_fill_innode_EP(EP10):
    node = Node(torch.rand(10, 10))
    EP10.innode = node
    assert AllEqual(EP10.nodes[0](), EP10.innode())


def test_feedforward_linear_EP(EP10):
    EP10.innode = Node(state=torch.rand(10, 10))
    EP10.feedforward(EP10.innode)
    torchseq = torch.nn.Sequential(*EP10.edges)
    result = torchseq(EP10.innode())
    assert (result == EP10.nodes[-1]()).all()


def test_feedforward_conv2d_CudaEP(CudaEP10Conv, FilledCudaNode):
    CudaEP10Conv.innode = FilledCudaNode
    CudaEP10Conv.feedforward(CudaEP10Conv.innode)
    torchseq = torch.nn.Sequential(*CudaEP10Conv.edges)
    result = torchseq(CudaEP10Conv.innode())
    assert (result == CudaEP10Conv.nodes[-1]()).all()


def test_dryrun_conv2d_CudaEP(CudaEP10Conv, FilledCudaNode):
    CudaEP10Conv.innode = FilledCudaNode
    CudaEP10Conv.feedforward(CudaEP10Conv.innode)
    CudaEP10Conv.resetnodes()
    torchseq = torch.nn.Sequential(*CudaEP10Conv.edges)
    result = torchseq(CudaEP10Conv.innode())
    assert (result != CudaEP10Conv.nodes[-1]()).all() and AllEqual(
        CudaEP10Conv.innode(), FilledCudaNode())


def test_cost_grad_EP():
    net = EP()
    t1 = torch.rand(10, 10)
    t1.requires_grad_()
    t2 = torch.rand(10, 10)
    t2.requires_grad_()
    costfunc = torch.nn.CrossEntropyLoss()
    net.outnode = Node(t2.detach().clone())
    net.addnode(Node(t1.detach().clone()))
    net.node_optim = torch.optim.SGD(net.nodes.parameters(), lr=1)

    net.nodes_step(net.cost())
    cost = costfunc(t1, t2)
    cost.backward()
    assert AllEqual(t1.grad, net.nodes[0]().grad)


@pytest.mark.xfail
def test_infer_EPConv(CudaEP10Conv, FilledCudaNode):
    CudaEP10Conv.node_optim = torch.optim.SGD(
        CudaEP10Conv.nodes.parameters(), 0.01)
    CudaEP10Conv.initall(FilledCudaNode.shape, torch.device("cuda"))
    print(CudaEP10Conv.energy())
    CudaEP10Conv.etol = 0.01
    out, elast, ediff = CudaEP10Conv.infer(FilledCudaNode.state, 200000)
    print(out, elast, ediff)
    assert (not AllEqual(out, 0)) and (ediff < CudaEP10Conv.etol)


def test_infer_EP(CudaEP10, VecCudaNode):
    CudaEP10.node_optim = torch.optim.SGD(CudaEP10.nodes.parameters(), 0.1)
    CudaEP10.initall(VecCudaNode.shape, torch.device("cuda"))
    CudaEP10.etol = 0.01
    out, elast, ediff = CudaEP10.infer(VecCudaNode.state, 100000)
    print(out, elast, ediff)
    assert (not AllEqual(out, 0)) and (ediff < CudaEP10.etol)
