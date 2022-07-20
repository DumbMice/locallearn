import torch
from torch.optim import Optimizer

def ggm_mean(measures):
    return sum(measures)/len(measures)

def ggm_inverse_average(measures):
    return len(measures)/sum(1./m for m in measures)

def ggm_median(measures):
    return measures[(len(measures))//2]

def blsgd(params,grads,lr,beta,measure_trace,measure_func,ggm_func,method_args):
    #TODO: delete legacy code
    measures = [measure_func(grad) for grad in grads]
    list(measure_trace[i].mul_(beta).add_(measures[i],alpha=1-beta) for i in range(len(measures)))
    global_measure = ggm_func(measure_trace)

    for i, param in enumerate(params):
        lr_i = lr/measure_trace[i]*global_measure
        param.add_(grads[i],alpha=-lr_i)


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
    method = {'SGD':torch.optim.SGD,'ASGD':torch.optim.ASGD, 'Adam':torch.optim.Adam}

    def __init__(self, params, lr=1e-3, measure='max',beta=0.9,gradient_get_method="average",method="SGD",method_args={}):
        if lr<0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >=0.0 ")
        if measure not in self.measures.keys():
            raise NameError(f"Invalid measure:{measure} - should be 'max', 'mean', 'std', or 'var'")
        if not 0.0<=beta<1.0:
            raise ValueError(f"Invalid beta parameter: {beta} - should be in [0.0,1.0)")
        if gradient_get_method not in self.ggm.keys():
            raise NameError(f"Invalid gradient_get_method:{gradient_get_method} - should be 'average', 'harmonic average', or 'median'")

        defaults = dict(lr=lr,measure=self.measures[measure],beta=beta,gradient_get_method=self.ggm[gradient_get_method],method=method,method_args=method_args)
        super(BalancedLayer,self).__init__(params,defaults)
        for group in self.param_groups:
            method = self.method[group['method']]
            group['method']=method([dict(params = p, lr=group['lr'],**group['method_args']) for p in group['params']])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for method_group in self.param_groups:
            measure_trace = []
            measure_func = method_group['measure']
            ggm_func = method_group['gradient_get_method']
            lr = method_group['lr']
            beta = method_group['beta']
            method = method_group['method']
            print(method)
            for group in method.param_groups:
                p = list(group['params'])[0]
                grad = p.grad
                if 'started' not in group:
                    # Exponential moving average of gradient values
                    group['measure_trace'] = measure_func(grad)
                    group['started']=True
                measure = measure_func(grad)
                group['measure_trace'].mul_(beta).add_(measure,alpha=1-beta)
                measure_trace.append(group['measure_trace'])
            global_measure = ggm_func(measure_trace)
            for group in method.param_groups:
                group['lr'] = lr/group['measure_trace']*global_measure
            method.step()


