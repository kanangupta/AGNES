import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required

class AGNES(Optimizer):

    def __init__(self, params, lr=1e-3, correction=0.1, momentum=0.99, weight_decay=0):
        defaults = dict(correction=correction, lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(AGNES, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AGNES, self).__setstate__(state)

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None: 
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = group['lr']
            momentum = group['momentum']
            correction = group['correction']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                state = self.state[p] #this contains the sequence of auxiliary variables we need: x'_n, v_n, v'_n
                if 'velocity' not in state:
                    state['velocity'] = torch.zeros_like(p.data) #initialize v_0 as zero
                vel = state['velocity']

                p.data.add_(d_p, alpha = -correction)
                vel.add_(d_p, alpha = -1)
                vel.mul_(momentum)
                p.data.add_(vel,alpha = lr)

        return loss