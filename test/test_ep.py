#!/usr/bin/env python


import pytest
from modules.node import *
from modules.edge import *
from .fixtures import *

def test_energy_gold_EP(MNISTCudaEP,MNISTgold,MNISTLoader):
    MNISTCudaEP.initall(input_shape=torch.Size([10,784]),device=torch.device('cuda'))
    MNISTCudaEP.activation = MNISTgold.phi[1]
    MNISTgold.to(torch.device('cuda'))
    MNISTgold.reset_state()
    trainldr,testldr = MNISTLoader
    trainiter = iter(trainldr)
    x,y = next(trainiter)
    x=x.view(-1,MNISTgold.dimensions[0]).cuda()
    y=y.cuda()
    MNISTgold.clamp_layer(0,x)
    MNISTgold.set_C_target(None)

    for i, u in enumerate(MNISTgold.u):
        MNISTCudaEP.nodes[i]().data=u.data.detach().clone()
    for i, W in enumerate(MNISTgold.W):
        MNISTCudaEP.edges[i].weight.data = W.weight.data.detach().clone()
        MNISTCudaEP.edges[i].bias.data = W.bias.data.detach().clone()
        print(MNISTCudaEP.edges[i].weight.data-W.weight.data)
        print(MNISTCudaEP.edges[i].bias.data-W.bias.data)
    MNISTCudaEP.clamp(0)
    MNISTgold.update_energy()
    print(f'gold is {MNISTgold.E} while mine is {MNISTCudaEP.energy()}')
    print(f'Initial states are:{MNISTCudaEP.nodes[0]()},{MNISTgold.u[0]}')
    MNISTCudaEP.beta=None
    assert AllEqual(MNISTgold.E,MNISTCudaEP.energy())<1e-3

def test_relax_gold_EP(MNISTCudaEP,MNISTgold,MNISTLoader):
    MNISTCudaEP.initall(input_shape=torch.Size([10,784]),device=torch.device('cuda'))
    MNISTgold.to(torch.device('cuda'))
    MNISTgold.reset_state()
    trainldr,testldr = MNISTLoader
    trainiter = iter(trainldr)
    x,y = next(trainiter)
    x=x.view(-1,MNISTgold.dimensions[0]).cuda()
    y=y.cuda()
    MNISTgold.clamp_layer(0,x)
    MNISTgold.set_C_target(None)

    for i, u in enumerate(MNISTgold.u):
        MNISTCudaEP.nodes[i].state=u.data.detach().detach().clone()
    for i, W in enumerate(MNISTgold.W):
        MNISTCudaEP.edges[i].weight.data = W.weight.data.detach().clone()
        MNISTCudaEP.edges[i].bias.data = W.bias.data.detach().clone()
    MNISTgold.update_energy()
    MNISTCudaEP.beta=None
    MNISTgold.clamp_layer(0,x)
    MNISTgold.u_relax(dt=0.1,n_relax=101,tol=0,tau=1.)
    MNISTCudaEP.node_optim = torch.optim.SGD(MNISTCudaEP.nodes.parameters(),lr=0.1)
    MNISTCudaEP.etol=0
    MNISTCudaEP.clamp(0)
    MNISTCudaEP.node_relax(lambda : torch.sum(MNISTCudaEP.energy()),100)
    print(f'Ending states are:{MNISTCudaEP.nodes[-1]()},{MNISTgold.u[-1]}')
    assert L2diff(MNISTCudaEP.nodes[-1](),MNISTgold.u[-1])<1e-3

def test_train_gold_EP(MNISTCudaEP,MNISTgold,MNISTLoader):
    beta =1

    def sq_error(u_last,target):
        loss = torch.nn.functional.mse_loss(u_last, target.float(), reduction='none')
        return beta * 0.5 * torch.sum(loss, dim=1)

    MNISTCudaEP.initall(input_shape=torch.Size([10,784]),device=torch.device('cuda'))
    MNISTgold.to(torch.device('cuda'))
    MNISTgold.reset_state()
    trainldr,testldr = MNISTLoader
    trainiter = iter(trainldr)

    MNISTCudaEP.node_optim = torch.optim.SGD(MNISTCudaEP.nodes.parameters(),lr=0.1)
    MNISTCudaEP.edge_optim = torch.optim.SGD(MNISTCudaEP.edges.parameters(),lr=0.1)
    MNISTCudaEP.etol=0
    w_optimizer = utils.create_optimizer(MNISTgold, 'sgd',  lr=0.1)

    for j in range(5):
        x,y = next(trainiter)
        x=x.view(-1,MNISTgold.dimensions[0]).cuda()
        y=y.cuda()

        MNISTgold.reset_state()

        MNISTgold.clamp_layer(0,x)
        # set EP equal to gold.
        for i, u in enumerate(MNISTgold.u):
            MNISTCudaEP.nodes[i].state=u.data.detach().detach().clone()
        for i, W in enumerate(MNISTgold.W):
            MNISTCudaEP.edges[i].weight.data = W.weight.data.detach().clone()
            MNISTCudaEP.edges[i].bias.data = W.bias.data.detach().clone()

        # gold reaches the 1st equilibrium
        MNISTgold.set_C_target(None)

        dE = MNISTgold.u_relax(dt=0.1,n_relax=100,tol=0,tau=1.)
        free_grads = MNISTgold.w_get_gradients()

        # add cost function
        # gold reaches the 2nd equilibrium
        MNISTgold.set_C_target(y)
        dE = MNISTgold.u_relax(dt=0.1,n_relax=100,tol=0,tau=1.)

        # gold computes grad and update weight.
        nudged_grads = MNISTgold.w_get_gradients()
        MNISTgold.w_optimize(free_grads, nudged_grads, w_optimizer)

        # EP reaches the 1st equilibrium
        MNISTCudaEP.clamp(0)
        MNISTCudaEP.beta=None
        MNISTCudaEP.node_relax(lambda : torch.sum(MNISTCudaEP.energy()),100)
        MNISTCudaEP.edge_optim.zero_grad()
        betaE = -1./beta*torch.mean(MNISTCudaEP.energy())
        betaE.backward(retain_graph=True)

        # EP reaches the 2nd equilibrium
        MNISTCudaEP.freeze()
        buff = MNISTCudaEP.edges[0].weight.grad.detach().clone()
        MNISTCudaEP.node_relax(lambda : torch.sum(MNISTCudaEP.energy()+beta*sq_error(MNISTCudaEP.nodes[-1](),y)),100)
        MNISTCudaEP.free()
        betaE = 1./beta*torch.mean(MNISTCudaEP.energy())
        betaE.backward(retain_graph=True)
        MNISTCudaEP.edge_optim.step()
        print(f'This is the {j}-th data')
        assert L2diff(MNISTgold.u[0],MNISTCudaEP.nodes[0]())<1e-3
        print(f'gold grad is {MNISTgold.W[1].weight.grad}')
        print(f'EP grad is {MNISTCudaEP.edges[1].weight.grad}')
        # assert not AllEqual(buff,MNISTCudaEP.edges[0].weight.grad)

    assert L2diff(MNISTgold.W[1].weight.grad,MNISTCudaEP.edges[1].weight.grad)<1e-3

def test_two_phase_update_lower_mse_loss_EP(MNISTCudaEP,MNISTLoader):

    def sq_error(u_last,target):
        loss = torch.nn.functional.mse_loss(u_last, target.float(), reduction='none')
        return 0.5 * torch.sum(loss, dim=1)

    MNISTCudaEP.initall(input_shape=torch.Size([10,784]),device=torch.device('cuda'))
    trainldr,testldr = MNISTLoader
    trainiter = iter(trainldr)

    MNISTCudaEP.node_optim = torch.optim.SGD(MNISTCudaEP.nodes.parameters(),lr=0.1)
    MNISTCudaEP.edge_optim = torch.optim.SGD(MNISTCudaEP.edges.parameters(),lr=0.01)
    MNISTCudaEP.etol=1e-3
    del MNISTCudaEP.costfunc
    MNISTCudaEP.costfunc = sq_error
    MNISTCudaEP.beta = 0.1
    MNISTCudaEP.max_iter = 200

    x,y = next(trainiter)
    x=x.view(x.shape[0],-1).cuda()
    y=torch.nn.functional.one_hot(y,num_classes=10).float().cuda()

    MNISTCudaEP.outnode = Node(state=y)
    pre_cost = torch.mean(MNISTCudaEP.cost())
    for i in range(5):
        MNISTCudaEP.two_phase_update(x,y)
        print(torch.mean(MNISTCudaEP.cost()))
    assert torch.mean(MNISTCudaEP.cost()-pre_cost)<0.

