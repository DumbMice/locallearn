#!/usr/bin/env python


import pytest
from modules.node import *
from modules.edge import *
from .fixtures import *
from itertools import cycle

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
    assert L2diff(MNISTgold.E,MNISTCudaEP.energy())<1e-4

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
    assert L2diff(MNISTCudaEP.nodes[-1](),MNISTgold.u[-1])<1e-4

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
    del MNISTCudaEP.costfunc
    MNISTCudaEP.costfunc = sq_error
    w_optimizer = utils.create_optimizer(MNISTgold, 'sgd',  lr=0.1)

    for j in range(5):
        x,y = next(trainiter)
        x=x.view(-1,MNISTgold.dimensions[0]).cuda()
        y=y.float().cuda()

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
        assert L2diff(MNISTgold.W[1].weight.grad,MNISTCudaEP.edges[1].weight.grad)<1e-4
        assert L2diff(MNISTgold.W[0].weight.grad,MNISTCudaEP.edges[0].weight.grad)<1e-4
        print(f'gold grad is {MNISTgold.W[1].weight.grad}')
        print(f'EP grad is {MNISTCudaEP.edges[1].weight.grad}')
        # assert not AllEqual(buff,MNISTCudaEP.edges[0].weight.grad)

    assert L2diff(MNISTgold.W[1].weight.grad,MNISTCudaEP.edges[1].weight.grad)<1e-4

@pytest.mark.skip
def test_two_phase_update_gold_EP(MNISTgold,MNISTCudaEP,MNISTLoader):
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
    MNISTCudaEP.beta = 1
    del MNISTCudaEP.costfunc
    MNISTCudaEP.costfunc = sq_error
    w_optimizer = utils.create_optimizer(MNISTgold, 'sgd',  lr=0.1)

    iters = 200
    for j in range(iters):
        x,y = next(trainiter)
        x=x.view(-1,MNISTgold.dimensions[0]).cuda()
        y=y.float().cuda()

        MNISTgold.reset_state()

        MNISTgold.clamp_layer(0,x)
        # set EP equal to gold.
        for i, u in enumerate(MNISTgold.u):
            MNISTCudaEP.nodes[i].state=u.data.detach().detach().clone()
        if j==0:
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
        MNISTCudaEP.two_phase_update(x,y,max_iter=100,etol=0.)


        print(f'This is the {j}-th data')
        assert L2diff(MNISTgold.u[0],MNISTCudaEP.nodes[0]())<1e-3
        assert L2diff(MNISTgold.W[1].weight.grad,MNISTCudaEP.edges[1].weight.grad)<1e-4
        assert L2diff(MNISTgold.W[0].weight.grad,MNISTCudaEP.edges[0].weight.grad)<1e-4
        print(f'gold grad is {MNISTgold.W[1].weight.grad}')
        print(f'EP grad is {MNISTCudaEP.edges[1].weight.grad}')

        if j != iters-1:
            MNISTgold.reset_state()
            MNISTgold.set_C_target(None)
            dE = MNISTgold.u_relax(dt=0.1,n_relax=100,tol=0,tau=1.)
            MNISTgold.set_C_target(y)
            print(f'gold test cost:{torch.mean(MNISTgold.c_energy.compute_energy(MNISTgold.u[-1]))}')
            for i, u in enumerate(MNISTgold.u):
                MNISTCudaEP.nodes[i].state=u.data.detach().detach().clone()
            out,e_last,e_diff = MNISTCudaEP.infer(x,reset=False,beta=0)
            test_cost = torch.mean(MNISTCudaEP.cost())
            print(f'EP test cost:{test_cost}')
            # assert not AllEqual(buff,MNISTCudaEP.edges[0].weight.grad)
    assert L2diff(MNISTgold.W[1].weight.grad,MNISTCudaEP.edges[1].weight.grad)<1e-4

@pytest.mark.skip
def test_two_phase_update_lower_mse_loss_EP(MNISTCudaEP,MNISTLoader100):

    def sq_error(u_last,target):
        loss = torch.nn.functional.mse_loss(u_last, target.float(), reduction='none')
        return 0.5 * torch.sum(loss, dim=1)

    MNISTCudaEP.initall(input_shape=torch.Size([100,784]),device=torch.device('cuda'))
    trainldr,testldr = MNISTLoader100
    trainiter = iter(trainldr)

    MNISTCudaEP.node_optim = torch.optim.SGD(MNISTCudaEP.nodes.parameters(),lr=0.05)
    MNISTCudaEP.edge_optim = torch.optim.ASGD(MNISTCudaEP.edges.parameters(),lr=0.1)
    MNISTCudaEP.etol=1e-3
    del MNISTCudaEP.costfunc
    MNISTCudaEP.costfunc = sq_error
    MNISTCudaEP.beta = 0.1
    MNISTCudaEP.max_iter = 2000

    x,y = next(trainiter)
    x=x.view(x.shape[0],-1).cuda()
    y=y.float().cuda()
    ybk = y.detach().clone()

    MNISTCudaEP.outnode = Node(state=y)
    out,_,_ = MNISTCudaEP.infer(x,reset=True)
    pre_cost = torch.mean(MNISTCudaEP.cost())
    print(f'pre cost is {pre_cost}')
    for i in range(500):
        MNISTCudaEP.two_phase_update(x,y)
        print(torch.mean(MNISTCudaEP.cost()-pre_cost))
        out,e_last,e_diff = MNISTCudaEP.infer(x,reset=True,beta=0)
        pre_cost = torch.mean(MNISTCudaEP.cost())
        assert AllEqual(MNISTCudaEP.costfunc(out,y),MNISTCudaEP.cost())
        print(f'out 1st:{out[0]}, y1st:{y[0]}, elast:{e_last}, e_diff:{e_diff},infer_cost:{pre_cost.item()}')
        assert AllEqual(y,ybk)
    assert torch.mean(MNISTCudaEP.cost()-pre_cost)<0.

@pytest.mark.skip
def test_two_phase_update_lower_crossentropy_loss_EP(MNISTCudaEP,MNISTLoader100):

    MNISTCudaEP.initall(input_shape=torch.Size([100,784]),device=torch.device('cuda'))
    trainldr,testldr = MNISTLoader100
    trainiter = iter(trainldr)

    MNISTCudaEP.node_optim = torch.optim.SGD(MNISTCudaEP.nodes.parameters(),lr=0.05)
    MNISTCudaEP.edge_optim = torch.optim.ASGD(MNISTCudaEP.edges.parameters(),lr=0.5)
    MNISTCudaEP.etol=1e-3
    MNISTCudaEP.beta = 1
    MNISTCudaEP.max_iter = 2000
    epoches = 10

    x,y = next(trainiter)
    x=x.view(x.shape[0],-1).cuda()
    y=y.float().cuda()

    MNISTCudaEP.outnode = Node(state=y)
    out,_,_ = MNISTCudaEP.infer(x,reset=True,beta=0)
    pre_cost = torch.mean(MNISTCudaEP.cost())
    print(f'pre cost is {pre_cost}')
    for epoch in range(epoches):
        trainiter = iter(trainldr)
        for i in range(599):
            x,y = next(trainiter)
            x=x.view(x.shape[0],-1).cuda()
            y=y.float().cuda()
            MNISTCudaEP.outnode = Node(state=y)
            if i%20==0:
                out,e_last,e_diff = MNISTCudaEP.infer(x,reset=True,beta=0)
                pre_cost = torch.mean(MNISTCudaEP.cost())
                print(f'step:{i},out 1st:{out[0]},y 1st:{y[0]}, elast:{e_last}, e_diff:{e_diff},infer_cost:{pre_cost.item()}')
                MNISTCudaEP.edge_optim.zero_grad()
                MNISTCudaEP.node_optim.zero_grad()
            MNISTCudaEP.two_phase_update(x,y)
        correct = 0
        total = 0
        out,e_last,e_diff = MNISTCudaEP.infer(x,reset=True,beta=0)
        # Compute test batch accuracy, energy and store number of seen batches
        correct += float(torch.sum(torch.argmax(out,1) == y.argmax(dim=1)))
        total += x.size(0)
        print(f'epoch {epoch} accuracy: {correct/total}')
        pre_cost = torch.mean(MNISTCudaEP.cost())
        MNISTCudaEP.edge_optim.zero_grad()
        MNISTCudaEP.node_optim.zero_grad()
    assert torch.mean(MNISTCudaEP.cost()-pre_cost)<0.

def test_two_phase_update_lower_crossentropy_loss_EP(MNISTCudaEP,MNISTLoader100):

    MNISTCudaEP.initall(input_shape=torch.Size([100,784]),device=torch.device('cuda'))
    trainldr,testldr = MNISTLoader100
    trainiter = iter(trainldr)

    MNISTCudaEP.node_optim = torch.optim.SGD(MNISTCudaEP.nodes.parameters(),lr=0.05)
    MNISTCudaEP.edge_optim = torch.optim.ASGD(MNISTCudaEP.edges.parameters(),lr=0.5)
    MNISTCudaEP.etol=1e-3
    MNISTCudaEP.beta = 1
    MNISTCudaEP.max_iter = 2000
    epoches = 10

    x,y = next(trainiter)
    x=x.view(x.shape[0],-1).cuda()
    y=y.float().cuda()

    MNISTCudaEP.outnode = Node(state=y)
    out,_,_ = MNISTCudaEP.infer(x,reset=True,beta=0)
    pre_cost = torch.mean(MNISTCudaEP.cost())
    print(f'pre cost is {pre_cost}')
    for epoch in range(epoches):
        trainiter = iter(trainldr)
        for i in range(599):
            x,y = next(trainiter)
            x=x.view(x.shape[0],-1).cuda()
            y=y.float().cuda()
            MNISTCudaEP.outnode = Node(state=y)
            if i%20==0:
                out,e_last,e_diff = MNISTCudaEP.infer(x,reset=True,beta=0)
                pre_cost = torch.mean(MNISTCudaEP.cost())
                print(f'step:{i},out 1st:{out[0]},y 1st:{y[0]}, elast:{e_last}, e_diff:{e_diff},infer_cost:{pre_cost.item()}')
                MNISTCudaEP.edge_optim.zero_grad()
                MNISTCudaEP.node_optim.zero_grad()
            MNISTCudaEP.two_phase_update(x,y)
        correct = 0
        total = 0
        out,e_last,e_diff = MNISTCudaEP.infer(x,reset=True,beta=0)
        # Compute test batch accuracy, energy and store number of seen batches
        correct += float(torch.sum(torch.argmax(out,1) == y.argmax(dim=1)))
        total += x.size(0)
        print(f'epoch {epoch} accuracy: {correct/total}')
        pre_cost = torch.mean(MNISTCudaEP.cost())
        MNISTCudaEP.edge_optim.zero_grad()
        MNISTCudaEP.node_optim.zero_grad()
    assert torch.mean(MNISTCudaEP.cost()-pre_cost)<0.

def test_three_phase_update_lower_crossentropy_loss_EP(MNISTCudaEP,MNISTLoader100):

    MNISTCudaEP.initall(input_shape=torch.Size([100,784]),device=torch.device('cuda'))
    trainldr,testldr = MNISTLoader100
    trainiter = iter(trainldr)

    MNISTCudaEP.node_optim = torch.optim.SGD(MNISTCudaEP.nodes.parameters(),lr=0.05)
    MNISTCudaEP.edge_optim = torch.optim.ASGD(MNISTCudaEP.edges.parameters(),lr=0.5)
    MNISTCudaEP.etol=1e-3
    MNISTCudaEP.beta = 0.5
    MNISTCudaEP.max_iter = 2000
    epoches = 10

    x,y = next(trainiter)
    x=x.view(x.shape[0],-1).cuda()
    y=y.float().cuda()

    MNISTCudaEP.outnode = Node(state=y)
    out,_,_ = MNISTCudaEP.infer(x,reset=True,beta=0)
    pre_cost = torch.mean(MNISTCudaEP.cost())
    print(f'pre cost is {pre_cost}')
    for epoch in range(epoches):
        trainiter = iter(trainldr)
        for i in range(599):
            x,y = next(trainiter)
            x=x.view(x.shape[0],-1).cuda()
            y=y.float().cuda()
            MNISTCudaEP.outnode = Node(state=y)
            if i%20==0:
                out,e_last,e_diff = MNISTCudaEP.infer(x,reset=True,beta=0)
                pre_cost = torch.mean(MNISTCudaEP.cost())
                print(f'step:{i},out 1st:{out[0]},y 1st:{y[0]}, elast:{e_last}, e_diff:{e_diff},infer_cost:{pre_cost.item()}')
                MNISTCudaEP.edge_optim.zero_grad()
                MNISTCudaEP.node_optim.zero_grad()
            MNISTCudaEP.three_phase_update(x,y)
        correct = 0
        total = 0
        out,e_last,e_diff = MNISTCudaEP.infer(x,reset=True,beta=0)
        # Compute test batch accuracy, energy and store number of seen batches
        correct += float(torch.sum(torch.argmax(out,1) == y.argmax(dim=1)))
        total += x.size(0)
        print(f'epoch {epoch} accuracy: {correct/total}')
        pre_cost = torch.mean(MNISTCudaEP.cost())
        MNISTCudaEP.edge_optim.zero_grad()
        MNISTCudaEP.node_optim.zero_grad()
    assert torch.mean(MNISTCudaEP.cost()-pre_cost)<0.
