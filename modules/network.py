#!/usr/bin/env python

import abc
import torch
from .node import *
from .edge import *

def identity(x):
    return x

class Network(abc.ABC, torch.nn.Module):

    """Network containing nodes and edges that has energy"""

    def __init__(
            self, innode=None, outnode=None,
            costfunc=torch.nn.CrossEntropyLoss(),
            etol=1e-1, max_iter=100):
        """TODO: to be defined. """
        super().__init__()
        self.nodes = torch.nn.ModuleList()
        self.freenodes_ = torch.nn.ModuleList()
        self.edges = torch.nn.ModuleList()
        self.costfunc = costfunc
        self.etol = etol
        self.max_iter = max_iter
        self.external_nodes = {'innode': None, 'outnode': None}
        self.extra_edges = []
        self.node_optim = None
        self.edge_optim = None

    def addnode(self, *args: type(Node)):
        """
        Add an edge to self.edges.
        """
        for node in args:
            self.nodes.append(node)
            self.freenodes_.append(node)

    def addclampednode(self, *args: type(Node)):
        """
        Add an edge to self.edges.
        """
        for node in args:
            self.nodes.append(node)

    def addcopynode(self, n, *args, **kwargs):
        """
        Add n nodes.
        """
        for i in range(n):
            self.addnode(Node(*args, **kwargs))

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
            edge: type(Edge),
            scale: float= 1.0):
        """
        Connect two nodes by an edge, and add them inside the network.
        """
        if prenode not in self.nodes:
            # TODO: Check if identical edges can be identified <13-04-22, Yang
            # Bangcheng> #
            if prenode.clamped:
                self.addnode(prenode)
            else:
                self.addclampednode(prenode)
        if posnode not in self.nodes:
            # TODO: Check if identical edges can be identified <13-04-22, Yang
            # Bangcheng> #
            if prenode.clamped:
                self.addnode(posnode)
            else:
                self.addclampednode(posnode)

        prenode.connectout.append(edge)
        posnode.connectin.append(edge)
        edge.connect(prenode, posnode)
        edge.scale = scale

        if not edge in self.edges:
            # TODO: Check if identical edges can be identified <13-04-22, Yang
            # Bangcheng> #
            self.addedge(edge)
        return edge

    @connect.register
    def _(self, prenode: int, posnode: int, edge: type(Edge),scale: float=1.0):
        self.nodes[prenode].connectout.append(edge)
        self.nodes[posnode].connectin.append(edge)
        edge.connect(self.nodes[prenode], self.nodes[posnode])
        edge.scale = scale

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

    def __repr__(self):
        return f'< {self.__class__.__mro__[2].__name__} {self.__class__.__mro__[1].__name__} {self.__class__.__mro__[0].__name__}(etol={self.etol}, max_iter={self.max_iter}, node_optim={self.node_optim}, edge_optim={self.edge_optim}, nodes={self.nodes}), edges={self.edges}, external_nodes={self.external_nodes}, extra_edges={self.extra_edges}, costfunc={self.costfunc} at {hex(id(self))}>'

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

    def nodes_step(self, e):
        """
        Gradient descent step of free energy on nodes
        """
        # TODO: Add Exception if optim not initiazlied <12-04-22,Yang
        # Bangcheng> #
        e.backward(inputs=list(self.freenodes_.parameters()))
        self.node_optim.step()

    def node_relax(self, e_func, max_iter=None, etol=None):
        """
        Multistep gradient descent of free energy on nodes until convergence before max iterations.
        """
        etol = etol if etol is not None else self.etol
        max_iter = max_iter if max_iter is not None else self.max_iter
        e_last = e_func()
        self.freeze()
        self.nodes_step(e_last)
        self.node_optim.zero_grad()
        e_diff = 0
        for step in range(max_iter-1):
            e = e_func()
            e_diff = abs(e-e_last)
            if e_diff < etol:
                self.node_optim.zero_grad()
                self.free()
                return e_diff, e_last
            e_last = e
            self.nodes_step(e)
            self.node_optim.zero_grad()
        self.free()
        return e_diff, e_last

    def edges_step(self, e):
        # TODO: Add Exception if optim not initiazlied <12-04-22,Yang
        # Bangcheng> #
        edgelist = list(self.edges.parameters())
        for a in (self.extra_edges):
            edgelist += list(a.parameters())
        e.backward(inputs=edgelist)
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

    def freeze(self, *args: int, extra_edges=[]):
        """
        Freeze edges by their indices inside self.edges
        """
        if len(args) == 0:
            for edge in self.edges:
                edge.freeze_grad()
        else:
            for ind in args:
                self.edges[ind].freeze_grad()
        for edge in self.extra_edges:
            edge.freeze_grad()

    def free(self, *args: int):
        """
        Free edges by their indices inside self.edges
        """
        if len(args) == 0:
            for edge in self.edges:
                edge.free_grad()
        else:
            for ind in args:
                self.edges[ind].free_grad()
        for edge in self.extra_edges:
            edge.free_grad()

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

    def feedforward(self, node, mem=None):
        if mem is None:
            mem = set()
        for edge in node.connectout:
            if edge not in mem:
                mem.add(edge)
                edge.pos.state = edge()
                self.feedforward(edge.pos,mem=mem)
        return mem

    @abc.abstractmethod
    def infer(self, input):
        """
        Given inputs, return outputs.
        """
        pass

    @abc.abstractmethod
    def initall(self):
        pass


class NToX(Network):
    def set_innode(self, nodes):
        if self.external_nodes['innode'] is None:
            self.external_nodes['innode'] = torch.nn.ModuleList(nodes)
        else:
            for i, node in enumerate(self.external_nodes['innode']):
                node.state = nodes[i].state.data
        for node in self.innode:
            node.clamp()
            node.activation = identity
            self.addclampednode(node)

    def extend_innode(self, nodes):
        if self.innode is None:
            self.set_innode(nodes)
            return
        else:
            self.external_nodes['innode'].extend(nodes)
        for node in nodes:
            node.clamp()
            node.activation = identity
            self.addclampednode(node)

    def initall(self, input_shapes, device=torch.device('cpu')):
        mem = set()
        for i, node in enumerate(self.innode):
            node = Node(torch.zeros(input_shapes[i]).to(device))
            mem = self.feedforward(self.innode, mem=mem)
        self.resetnodes()


class OneToX(Network):
    def set_innode(self, node):
        if self.external_nodes['innode'] is None:
            self.external_nodes['innode'] = node
            self.addclampednode(node)
        else:
            self.external_nodes['innode'].state = node.state.data
        self.external_nodes['innode'].clamp()
        self.external_nodes['innode'].activation = identity

    def initall(self, input_shape, device=torch.device('cpu')):
        self.innode = Node(torch.zeros(input_shape).to(device))
        self.feedforward(self.innode)
        self.resetnodes()


class XToN(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(costfunc={}, *args, **kwargs)

    def set_outnode(self, nodes):
        if self.external_nodes['outnode'] is None:
            self.external_nodes['outnode'] = torch.nn.ModuleList([nodes])
        else:
            for i, node in enumerate(self.external_nodes['outnode']):
                node.state = nodes[i].state.data
        for node in self.outnode:
            node.clamp()

    def extend_outnode(self, nodes):
        if self.outnode is None:
            self.set_outnode(nodes)
        else:
            self.external_nodes['outnode'].extend(nodes)
            for node in nodes:
                node.clamp()

    def cost(self):
        # TODO: Add support for distince cost func <26-04-22, Yang Bangcheng> #
        return sum(
            self.costfunc[key](*list(node() for node in key))
            for key in self.costfunc.keys())


class XToOne(Network):
    def set_outnode(self, node):
        if self.external_nodes['outnode'] is None:
            self.external_nodes['outnode'] = node
        else:
            self.external_nodes['outnode'].state = node.state.data
        self.external_nodes['outnode'].clamp()
        self.external_nodes['outnode'].activation = identity

    def cost(self):
        return self.costfunc(
            self.nodes[-1].state, self.outnode.state)


class OneToOne(OneToX, XToOne):
    def infer(self, input, reset=False,
              mean=True, max_iter=None, etol=None, beta=None):
        max_iter = max_iter if max_iter is not None else self.max_iter
        self.innode = Node(state=input)
        if reset:
            self.resetnodes()
        self.node_optim.zero_grad()
        Ediff = self.node_relax(
            lambda: torch.sum(
                self.energy(beta)),
            max_iter=max_iter, etol=etol)[0].item()

        Elast = torch.mean(
            self.energy()).item() if mean else torch.sum(
            self.energy()).item()
        return self.nodes[-1].state.data, Elast, Ediff

    def addlayerednodes(self, n, outnode=True, *args, **kwargs):
        """
        Add n nodes (including innode). With outnode=true, create one extra outnode.
        """
        innode = Node(*args, **kwargs)
        self.innode = innode
        self.addcopynode(n-1, *args, **kwargs)
        if outnode:
            self.outnode = Node()


class EP(OneToOne):

    """Docstring for EP. """

    def __init__(self, beta=None, *args, **kwargs):
        """TODO: to be defined. """
        super().__init__(*args, **kwargs)
        self.beta = beta

    def energy(self, beta=None):
        C = 0
        beta = beta if beta is not None else self.beta
        if beta is not None and beta != 0:
            C = beta*self.cost()
        E = 0
        for i, node in enumerate(self.nodes):
            if node.energy() is not None:
                E += node.energy()
            else:
                self_energy = 0.5*torch.sum((node()
                                            ** 2).flatten(start_dim=1), 1)
                E += self_energy

        for i, edge in enumerate(self.edges):
            if edge.energy() is not None:
                E -= edge.scale*edge.energy()
            else:
                product = edge(edge.pre.activate())*edge.pos.activate()
                interaction = torch.sum(product.flatten(start_dim=1), 1)
                E -= edge.scale*interaction
        return E+C

    def two_phase_update(self, x, y, max_iter=None, etol=None,repeat=1,beta=0):
        self.outnode = Node(state=y)
        self.outnode.clamp()
        self.infer(x, reset=True, max_iter=max_iter, etol=etol, beta=0)
        # EP reaches first equilibrium
        self.edge_optim.zero_grad()
        edgelist = list(self.edges.parameters())
        for a in (self.extra_edges):
            edgelist += list(a.parameters())
        torch.mean(-1./self.beta*self.energy(beta=beta)).backward(inputs=edgelist)
        _, elast, ediff = self.infer(
            x, max_iter=max_iter, etol=etol, reset=False)
        self.edges_step(torch.mean(1./self.beta*self.energy(beta=-beta)))
        for i in range(repeat-1):
            self.infer(x, reset=False, max_iter=max_iter, etol=etol, beta=0)
            # EP reaches first equilibrium
            self.edge_optim.zero_grad()
            torch.mean(-1./self.beta*self.energy(beta=beta)).backward(inputs=edgelist)
            _, elast, ediff = self.infer(
                x, max_iter=max_iter, etol=etol, reset=False)
            self.edges_step(torch.mean(1./self.beta*self.energy(beta=-beta)))

    def three_phase_update(self, x, y, max_iter=None, etol=None,repeat=1,beta=0):
        self.outnode = Node(state=y)
        self.outnode.clamp()
        self.infer(x, reset=True, max_iter=max_iter, etol=etol, beta=0)
        # EP reaches first equilibrium
        self.infer(
            x,
            max_iter=max_iter,
            etol=etol,
            reset=False,
            beta=self.beta)
        self.edge_optim.zero_grad()
        edgelist = list(self.edges.parameters())
        for a in (self.extra_edges):
            edgelist += list(a.parameters())
        torch.mean(0.5/self.beta*self.energy(beta=beta)).backward(inputs=edgelist)
        _, elast, ediff = self.infer(
            x,
            max_iter=max_iter,
            etol=etol,
            reset=False,
            beta=-self.beta)
        self.edges_step(torch.mean(-0.5/self.beta*self.energy(beta=-beta)))
        for i in range(repeat-1):
            self.infer(x, reset=False, max_iter=max_iter, etol=etol, beta=0)
            # EP reaches first equilibrium
            self.infer(
                x,
                max_iter=max_iter,
                etol=etol,
                reset=False,
                beta=self.beta)
            self.edge_optim.zero_grad()
            torch.mean(0.5/self.beta*self.energy(beta=beta)).backward(inputs=edgelist)
            _, elast, ediff = self.infer(
                x,
                max_iter=max_iter,
                etol=etol,
                reset=False,
                beta=-self.beta)
            self.edges_step(torch.mean(-0.5/self.beta*self.energy(beta=-beta)))


class PC(OneToOne):

    def __init__(self, beta=None, *args, **kwargs):
        """TODO: to be defined. """
        super().__init__(*args, **kwargs)
        self.beta = beta

    def energy(self, beta=None):
        C = 0
        beta = beta if beta is not None else self.beta
        if beta is not None and beta != 0:
            C = beta*self.cost()
        E = 0.
        for i, edge in enumerate(reversed(self.edges)):
            if edge.energy() is not None:
                E += edge.scale*edge.energy()
            else:
                E += edge.scale*torch.sum(((edge.pos()-edge()) **
                               2).flatten(start_dim=1), 1)
        return E+C

    def one_phase_update(self, x, y, max_iter=None, etol=None, beta=None):
        self.outnode = Node(state=y)
        self.outnode.clamp()
        self.innode = Node(state=x)
        self.feedforward(self.innode)
        _, elast, ediff = self.infer(
            x, reset=False, max_iter=max_iter, etol=etol, beta=beta)
        self.edge_optim.zero_grad()
        self.edges_step(torch.mean(self.energy(beta=beta)))
        return elast, ediff


class NEP(NToX, EP):

    def infer(self, inputs, reset=False,
              mean=True, max_iter=None, etol=None, beta=None):
        max_iter = max_iter if max_iter is not None else self.max_iter
        # self.freeze()
        self.innode = [Node(state=input) for input in inputs]
        if reset:
            self.resetnodes()
        self.node_optim.zero_grad()
        Ediff = self.node_relax(
            lambda: torch.sum(
                self.energy(beta)),
            max_iter=max_iter, etol=etol)[0].item()
        Elast = torch.mean(
            self.energy()).item() if mean else torch.sum(
            self.energy()).item()
        # self.free()
        return self.nodes[-1].state.data, Elast, Ediff


class PCN(XToN, PC):
    pass

class BiPC(PCN):
    def __init__(self, beta=None, reverse_scale=1e-6, *args, **kwargs):
        """TODO: to be defined. """
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.reverse_scale = reverse_scale

    def energy(self,beta=None):
        C=0.
        beta=beta if beta is not None else self.beta
        if beta is not None and beta !=0:
            # C=beta*(self.cost()+self.cost_b())
            C = beta*self.cost()
        #正反向energy
        E=0.
        for edge in reversed(self.edges):
            if edge.energy() is not None:
                E+=edge.energy()
            else:
                E+=torch.sum(((edge.pos()-edge())**2).flatten(start_dim=1),1)
                E+=self.reverse_scale*torch.sum(((edge.pre()-edge.reverse(edge.pos()))**2).flatten(start_dim=1),1)
        return E+C
