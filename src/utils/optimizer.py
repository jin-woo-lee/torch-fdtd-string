import sys
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, model_size=None):
        self.warmup_steps = warmup_steps

        self.model_size = model_size

        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        if self.model_size is None:
            scale = scale * self.warmup_steps ** (0.5)
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            return [self.model_size ** (-0.5) * scale for _ in self.base_lrs]

 
class Novograd(torch.optim.Optimizer):
    """
    Implements Novograd algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging: gradient averaging
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.95, 0), eps=1e-8,
                 weight_decay=0, grad_averaging=False, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                      weight_decay=weight_decay,
                      grad_averaging=grad_averaging,
                      amsgrad=amsgrad)

        super(Novograd, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Novograd, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Sparse gradients are not supported.')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros([]).to(state['exp_avg'].device)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros([]).to(state['exp_avg'].device)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                norm = torch.sum(torch.pow(grad, 2))

                if exp_avg_sq == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(norm, alpha=1 - beta2)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                grad.div_(denom)
                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])
                if group['grad_averaging']:
                    grad.mul_(1 - beta1)
                exp_avg.mul_(beta1).add_(grad)

                p.data.add_(exp_avg, alpha=-group['lr'])
        
        return loss


def get_optimizer(optimizer_name, model_parameters, config):

    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_parameters, **config)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_parameters, **config)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model_parameters, **config)
    elif optimizer_name == "radam":
        optimizer = torch.optim.RAdam(model_parameters, **config)
    elif optimizer_name == "novograd":
        optimizer = Novograd(model_parameters, **config)
    else:
        print('Unknown optimizer', optimizer_name)
        sys.exit()
    return optimizer

def get_scheduler(scheduler_name, optimizer, config):
    if scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config)
    elif scheduler_name == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config)
    elif scheduler_name == 'sgdr':
        scheduler = SGDRScheduler(optimizer, **config)
    elif scheduler_name == 'lambda_lr':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, **config)
    elif scheduler_name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config)
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **config)
    elif scheduler_name == 'noam':
        scheduler = NoamLR(optimizer, **config)
    elif scheduler_name == 'constant':
        scheduler = None
    else:
        print('Unknown scheduler', scheduler_name)
        sys.exit()
    return scheduler


