#!/usr/bin/env python

import abc
import torch
from .node import *
from .edge import *


class Network(abc.ABC, torch.nn.Module):

    """Network containing nodes and edges that has energy"""

    def __init__(
            self, innode=None, outnode=None,
            costfunc=torch.nn.CrossEntropyLoss(),
            etol=1e-1):
        """TODO: to be defined. """
        super().__init__()
        self.nodes = torch.nn.ModuleList()
        self.edges = torch.nn.ModuleList()
        self.costfunc = costfunc
        self.etol = etol
        self.external_nodes = {'innode': innode, 'outnode': outnode}
        self.node_optim = None
        self.edge_optim = None

    def addnode(self, *args: type(Node)):
        """
        Add an edge to self.edges.
        """
        for node in args:
            self.nodes.append(node)

    def addedge(self, *args: type(Edge)):
        """
        Add an edge to self.edges.
        """
        for edge in args:
            self.edges.append(edge)

    @singledispatchmethod
    def connect(
            self, prenode: type(Node),
            posnode: type(Node),
            edge: type(Edge)):
        """
        Connect two nodes by an edge, and add them inside the network.
        """
        if prenode not in self.nodes:
            # TODO: Check if identical edges can be identified <13-04-22, Yang
            # Bangcheng> #
            self.addnode(prenode)
        if not posnode in self.nodes:
            # TODO: Check if identical edges can be identified <13-04-22, Yang
            # Bangcheng> #
            self.addnode(posnode)

        prenode.connectout.append(edge)
        posnode.connectin.append(edge)
        edge.connect(prenode, posnode)

        if not edge in self.edges:
            # TODO: Check if identical edges can be identified <13-04-22, Yang
            # Bangcheng> #
            self.addedge(edge)
        return edge

    @connect.register
    def _(self, prenode: int, posnode: int, edge: type(Edge)):
        self.nodes[prenode].connectout.append(edge)
        self.nodes[posnode].connectin.append(edge)
        edge.connect(self.nodes[prenode], self.nodes[posnode])

        if not edge in self.edges:
            # TODO: Check if identical edges can be identified <13-04-22, Yang
            # Bangcheng> #
            self.addedge(edge)
        return edge

    @abc.abstractmethod
    def cost(self):
        """
        Calculate cost between outnode and some other node inside the network.
        """
        pass

    @abc.abstractmethod
    def energy(self):
        """
        Calculate free energy of the current network.
        """
        pass

    def __setattr__(self, name, value):
        """
        Overwrite __setattr__ to bypass default setattr behavior of torch.nn.Module
        on innode and outnode
        """
        if name == 'innode':
            self.set_innode(value)
        elif name == 'outnode':
            self.set_outnode(value)
        else:
            super().__setattr__(name, value)

    @property
    def innode(self):
        """
        Input node, always contained inside self.nodes.
        """
        return self.external_nodes['innode']

    @abc.abstractmethod
    def set_innode(self, val):
        """
        Set one node to be the innode. By default, the innode should be self.nodes[0].
        """
        pass

    @property
    def outnode(self):
        """
        External node, excluded from self.nodes.
        """
        return self.external_nodes['outnode']

    @abc.abstractmethod
    def set_outnode(self, val):
        """
        Set one node to be the innode. By default, the innode should not be within self.nodes.
        """
        pass

    @property
    def batch(self):
        return self.innode.batch

    def addlayerednodes(self, n, outnode=True, *args, **kwargs):
        """
        Add n nodes (including innode). With outnode=true, create one extra outnode.
        """
        innode = Node(*args, **kwargs)
        self.innode = innode
        for i in range(n-1):
            self.addnode(Node(*args, **kwargs))
        if outnode:
            self.outnode = Node()

    def nodes_step(self, e):
        """
        Gradient descent step of free energy on nodes
        """
        # TODO: Add Exception if optim not initiazlied <12-04-22,Yang
        # Bangcheng> #
        e.backward(retain_graph=True)
        self.node_optim.step()

    def node_relax(self, e_func, max_iter=100):
        """
        Multistep gradient descent of free energy on nodes until convergence before max iterations.
        """
        e_last = e_func()
        self.nodes_step(e_last)
        self.node_optim.zero_grad()
        e_diff = 0
        for step in range(max_iter-1):
            e = e_func()
            e_diff = abs(e-e_last)
            print(step,e_diff)
            if e_diff < self.etol:
                self.node_optim.zero_grad()
                return e_diff, e_last
            e_last = e
            self.nodes_step(e)
            self.node_optim.zero_grad()
        return e_diff, e_last

    def edges_step(self, e):
        # TODO: Add Exception if optim not initiazlied <12-04-22,Yang
        # Bangcheng> #
        e.backward()
        self.edge_optim.step()

    def clamp(self, *args: int):
        """
        Clamp nodes by their indices inside self.nodes
        """
        if len(args) == 0:
            for node in self.nodes:
                node.clamp()
        else:
            for ind in args:
                self.nodes[ind].clamp()

    def unclamp(self, *args: int):
        """
        Unclamp nodes by their indices inside self.nodes
        """
        if len(args) == 0:
            for node in self.nodes:
                node.unclamp()
        else:
            for ind in args:
                self.nodes[ind].unclamp()

    def freeze(self, *args: int):
        """
        Freeze edges by their indices inside self.edges
        """
        if len(args) == 0:
            for edge in self.edges:
                edge.freeze()
        else:
            for ind in args:
                self.edges[ind].freeze()

    def free(self, *args: int):
        """
        Free edges by their indices inside self.edges
        """
        if len(args) == 0:
            for edge in self.edges:
                edge.free()
        else:
            for ind in args:
                self.edges[ind].free()

    def initnodes(self, allow_initialized=True, *args: int, **kwargs):
        if len(args) == 0:
            for node in self.nodes:
                if (not isinstance(
                        node.state, torch.nn.parameter.UninitializedParameter) and allow_initialized):
                    continue
                node.init_data(**kwargs)
        else:
            for ind in args:
                if (not isinstance(
                        node.state, torch.nn.parameter.UninitializedParameter) and allow_initialized):
                    continue
                self.nodes[ind].init_data(**kwargs)

    def resetnodes(self, *args: int, **kwargs):
        if len(args) == 0:
            for node in self.nodes:
                node.reset(batch=self.batch, **kwargs)
        else:
            for ind in args:
                self.nodes[ind].reset(batch=self.batch, **kwargs)

    @abc.abstractmethod
    def feedforward(self, node):
        """
        Pass data of innode through edges to fill the network.
        """
        pass

    def initall(self,input_shape,device=torch.device("cpu")):
        self.innode = Node(torch.zeros(input_shape).to(device))
        self.feedforward(self.innode)
        self.resetnodes()

class EP(Network):

    """Docstring for EP. """

    def __init__(self, beta=None, *args, **kwargs):
        """TODO: to be defined. """
        super().__init__(*args, **kwargs)
        self.beta = beta

    def set_innode(self, node):
        if self.external_nodes['innode'] is None:
            self.external_nodes['innode'] = node
            self.addnode(node)
        else:
            self.external_nodes['innode'].state = node.state.data
        node.clamp()
        node.activation = lambda x: x

    def set_outnode(self, node):
        if self.external_nodes['outnode'] is None:
            self.external_nodes['outnode'] = node
        else:
            self.external_nodes['outnode'].state = node.state.data
        node.clamp()

    def feedforward(self, node):
        for edge in node.connectout:
            edge.pos.state = edge()
            self.feedforward(edge.pos)

    def energy(self):
        C = 0
        if self.beta is not None:
            C = self.beta*self.cost()
        E = 0
        for node in self.nodes:
            E += 0.5*torch.sum((node()**2).flatten(start_dim=1), 1)

        for i,edge in enumerate(self.edges):
            # TODO: Handling Convolution layers <18-04-22, Yang Bangcheng> #
            E -= torch.sum((torch.nn.functional.linear(edge.pre.activate(),edge.weight)*edge.pos.activate()).flatten(start_dim=1), 1)
            E -= torch.einsum('i,ji->j',edge.bias,(edge.pos.activate()))
        return E+C

    def cost(self):
        return self.costfunc(
            input=self.nodes[-1].state, target=self.outnode.state)

    def infer(self, input, max_iter=100, requires_grad=True, mean=False):
        if not requires_grad:
            self.freeze()
        self.innode = Node(state=input)
        self.node_optim.zero_grad()
        Ediff = self.node_relax(
            lambda: torch.sum(
                self.energy()),
            max_iter)[0].item()
        Elast = torch.mean(
            self.energy()).item() if mean else torch.sum(
            self.energy()).item()
        if not requires_grad:
            self.free()
        return self.nodes[-1].state.data, Elast, Ediff
