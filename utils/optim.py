import torch
from torch.optim import Optimizer

def ggm_mean(measures):
    return sum(measures)/len(measures)

def ggm_inverse_average(measures):
    return len(measures)/sum(1./m for m in measures)

def ggm_median(measures):
    return measures[(len(measures))//2]

class BalancedLayer(Optimizer):
    """ Implements Balancing Layer learning rate optmization algorithm

    Parameters:
        lr(float): learning rate. Default 1e-3
        measure(string): balance strategy, max, mean, std or var. Default max.
        beta(flaot): decay coefficient within time window. Default 0.9.
        gradient_get_method(string): method to get general gradient, average, harmonic average, median. Default average.
    """

    measures = {'max':torch.max, 'mean':torch.mean, 'std':torch.std, 'var':lambda x: torch.std(x)**2}
    ggm = {'average':ggm_mean, 'harmonic average':ggm_inverse_average, 'median':ggm_median}

    def __init__(self, params, lr=1e-3, measure='max',beta=0.9,gradient_get_method="average"):
        if lr<0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >=0.0 ")
        if measure not in self.measures.keys():
            raise NameError(f"Invalid measure:{measure} - should be 'max', 'mean', 'std', or 'var'")
        if not 0.0<=beta<1.0:
            raise ValueError(f"Invalid beta parameter: {beta} - should be in [0.0,1.0)")
        if gradient_get_method not in self.ggm.keys():
            raise NameError(f"Invalid gradient_get_method:{gradient_get_method} - should be 'average', 'harmonic average', or 'median'")
        defaults = dict(lr=lr,measure=self.measures[measure],beta=beta,gradient_get_method=self.ggm[gradient_get_method])
        super(BalancedLayer,self).__init__(params,defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            measure_trace = []
            params_with_grad = []
            grads = []
            measure_func = group['measure']
            ggm_func = group['gradient_get_method']
            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('BalancedLayer does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['measure_trace'] = measure_func(grad)
                measure_trace.append(state['measure_trace'])
                state['step']+=1
            lr = group['lr']
            beta = group['beta']

            blgd(params_with_grad,grads,lr,beta,measure_trace,measure_func,ggm_func)


def blgd(params,grads,lr,beta,measure_trace,measure_func,ggm_func):
    measures = [measure_func(grad) for grad in grads]
    global_measure = ggm_func(measures)

    for i, param in enumerate(params):
        measure_trace[i].mul_(beta).add_(measures[i],alpha=1-beta)
        lr_i = lr/measure_trace[i]*global_measure
        param.add_(grads[i],alpha=-lr_i)

