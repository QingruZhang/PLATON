# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import math
import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from pytorch_pretrained_bert.optimization import warmup_constant, warmup_cosine, warmup_linear
from typing import Callable, Iterable, Tuple

def warmup_linear_xdl(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return (1.0 - x)/(1.0 - warmup)

def schedule_func(sch):
    try:
        f = eval(sch)
    except:
        f = warmup_linear
    return f

class Adamax(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    by xiaodl
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        self.init_sensitivity_record()
        super(Adamax, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_inf'].to(device)

    def initialize_step(self, initial_step):
        """Initialize state with a defined step (but we don't have stored averaged).
        Arguments:
            initial_step (int): Initial step number.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # State initialization
                state['step'] = initial_step
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_inf'] = torch.zeros_like(p.data)

    def init_sensitivity_record(self):
        self.modulated_lrs = {}
        self.exp_avg_ipts = {}
        self.ipts = {}

    def update_sensitivity_record(self, key, modulated_lr, ipt):
        if key not in self.modulated_lrs:
            self.modulated_lrs[key] = []
            self.ipts[key] = []
        self.modulated_lrs[key].append(modulated_lr)
        self.ipts[key].append(ipt)

    def get_sensitivity_record(self):
        return self.modulated_lrs, self.exp_avg_ipts, self.ipts

    def step(self, closure=None, update_record=False):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_inf'] = torch.zeros_like(p.data)

                exp_avg, exp_inf = state['exp_avg'], state['exp_inf']
                beta1, beta2 = group['betas']
                eps = group['eps']
                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # Update the exponentially weighted infinity norm.
                norm_buf = torch.cat([
                    exp_inf.mul_(beta2).unsqueeze(0),
                    grad.abs().add_(eps).unsqueeze_(0)
                ], 0)
                torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))
                update = exp_avg / (exp_inf + eps)

                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                ###################################
                if update_record:
                    ipt = (p.data * grad).abs()
                    self.update_sensitivity_record((group['weight_decay'], i), lr_scheduled, ipt.clone().data)
                ###################################

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)
                state['step'] += 1

        return loss

class RAdam(Optimizer):
    """Modified from: https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 betas=(0.9, 0.999), eps=1e-6, weight_decay=0.001,
                 max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_avg_sq'].to(device)

    def initialize_step(self, initial_step):
        """Initialize state with a defined step (but we don't have stored averaged).
        Arguments:
            initial_step (int): Initial step number.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # State initialization
                state['step'] = initial_step
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # set_trace()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                eps = group['eps']
                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                state['step'] += 1

                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = lr_scheduled * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = lr_scheduled / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * lr_scheduled, p_data_fp32)

                p.data.copy_(p_data_fp32)

        return loss

class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in `Decoupled Weight Decay Regularization
    <https://arxiv.org/abs/1711.05101>`__.
    Parameters:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether ot not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        warmup: float = -1,
        t_total: int = -1,
        schedule: str = 'warmup_linear',
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        max_grad_norm: float = 1.0
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, warmup=warmup, t_total=t_total, schedule=schedule, \
                        betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super(AdamW, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_avg_sq'].to(device)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss

class StructAwareAdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in `Decoupled Weight Decay Regularization
    <https://arxiv.org/abs/1711.05101>`__.
    Parameters:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether ot not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        warmup: float = -1,
        t_total: int = -1,
        schedule: str = 'warmup_linear',
        betas: Tuple[float, float] = (0.9, 0.999, 0.9),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        max_grad_norm: float = 1.0
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[2]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, warmup=warmup, t_total=t_total, schedule=schedule, \
                        betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super(StructAwareAdamW, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_avg_sq'].to(device)
            state['exp_avg_head_ipt'].to(device)
            state['exp_avg_ffn_ipt'].to(device)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            beta1, beta2, beta3 = group["betas"]

            if group['params_type'] == "head":
                head_ipts, exp_avg_head_ipts = [], []
                for p in group['ipts_params']:
                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        # Exponential moving average importance score of head
                        state['exp_avg_head_ipt'] = torch.zeros_like(p.data)
                    exp_avg_head_ipt = state['exp_avg_head_ipt']
                    exp_avg_head_ipt.mul_(beta3).add_(p.grad.abs(), alpha=1.0 - beta3)
                    head_ipts.append(p.grad.abs())
                    exp_avg_head_ipts.append(exp_avg_head_ipt)

            if group['params_type'] == "ffn":
                ffn_ipts, exp_avg_ffn_ipts = [], []
                for p in group['ipts_params']:
                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        # Exponential moving average importance score of ffn
                        state['exp_avg_ffn_ipt'] = torch.zeros_like(p.data)
                    exp_avg_ffn_ipt = state['exp_avg_ffn_ipt']
                    exp_avg_ffn_ipt.mul_(beta3).add_(p.grad.abs(), alpha=1.0 - beta3)
                    ffn_ipts.append(p.grad.abs())
                    exp_avg_ffn_ipts.append(exp_avg_ffn_ipt)

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                bias_correction1, bias_correction2, bias_correction3 = 1, 1, 1
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    bias_correction3 = 1.0 - beta3 ** state["step"]
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                if group['params_type'] == "head":
                    if group["weight_decay"] > 0.0:
                        layer_id = i//4
                    else:
                        layer_id = i//3
                    exp_avg_head_ipt = exp_avg_head_ipts[layer_id]
                    head_ipt = head_ipts[layer_id]
                    if group["weight_decay"] > 0.0 and i%4 == 3:
                        step_size = step_size * torch.ones_like(p.data).view(p.data.shape[0], head_ipt.shape[0], -1) \
                                              * ((head_ipt - exp_avg_head_ipt / bias_correction3).abs() / (head_ipt + group["eps"]))[None, :, None]
                    else:
                        step_size = step_size * torch.ones_like(p.data).view(head_ipt.shape[0], -1) \
                                              * ((head_ipt - exp_avg_head_ipt / bias_correction3).abs() / (head_ipt + group["eps"]))[:, None]
                    p.data = p.data - step_size.view(p.data.shape) * (exp_avg / denom)

                    if group["weight_decay"] > 0.0:
                        decay_lr = group["lr"] * torch.ones_like(p.data).view(p.data.shape[0], head_ipt.shape[0], -1) \
                                               * ((head_ipt - exp_avg_head_ipt / bias_correction3).abs() / (head_ipt + group["eps"]))[None, :, None]
                        p.data = p.data - decay_lr.view(p.data.shape) * group["weight_decay"]
                elif group['params_type'] == "ffn":
                    layer_id = i//2
                    exp_avg_ffn_ipt = exp_avg_ffn_ipts[layer_id]
                    ffn_ipt = ffn_ipts[layer_id]
                    step_size = step_size * ((ffn_ipt - exp_avg_ffn_ipt / bias_correction3).abs() / (ffn_ipt + group["eps"])).item()
                    decay_lr = group["lr"] * ((ffn_ipt - exp_avg_ffn_ipt / bias_correction3).abs() / (ffn_ipt + group["eps"])).item()
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                    if group["weight_decay"] > 0.0:
                        p.data.add_(p.data, alpha=-decay_lr * group["weight_decay"])
                else:
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                    if group["weight_decay"] > 0.0:
                        p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                # if group["weight_decay"] > 0.0:
                #     p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss

class StructAwareAdamax(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    by xiaodl
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 betas=(0.9, 0.999), beta3=0.99, eps=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        betas = betas + (beta3,)
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[2]))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        self.init_sensitivity_record()
        super(StructAwareAdamax, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def init_sensitivity_record(self):
        self.actual_lr = []
        self.modulated_lr = []
        self.ipts = []
        self.exp_avg_ipts = []

    def update_sensitivity_record(self, actual_lr, modulated_lr, ipts, exp_avg_ipts):
        self.actual_lr.append(actual_lr)
        self.modulated_lr.append(modulated_lr)
        self.ipts.append(ipts)
        self.exp_avg_ipts.append(exp_avg_ipts)

    def get_sensitivity_record(self):
        return self.actual_lr, self.modulated_lr, self.ipts, self.exp_avg_ipts

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_inf'].to(device)
            state['exp_avg_head_ipt'].to(device)
            state['exp_avg_ffn_ipt'].to(device)

    def step(self, closure=None, update_record=False):
        loss = None
        if closure is not None:
            loss = closure()

        if update_record:
            actual_lr, modulated_lr, ipts, exp_avg_ipts = {}, {}, {}, {}

        for group in self.param_groups:

            beta1, beta2, beta3 = group["betas"]

            if group['params_type'] == "head":
                head_ipts, exp_avg_head_ipts = [], []
                # ipt_mean = sum(sum([p.grad.abs() for p in group['ipts_params']])) / (len(group['ipts_params'])*group['ipts_params'][0].shape[0])
                for p in group['ipts_params']:

                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        # Exponential moving average importance score of head
                        state['exp_avg_head_ipt'] = torch.zeros_like(p.data)
                    exp_avg_head_ipt = state['exp_avg_head_ipt']
                    exp_avg_head_ipt.mul_(beta3).add_(p.grad, alpha=1.0 - beta3)
                    head_ipts.append(p.grad)
                    exp_avg_head_ipts.append(exp_avg_head_ipt)

            if group['params_type'] == "ffn":
                ffn_ipts, exp_avg_ffn_ipts = [], []
                # ipt_mean = sum([p.grad.abs() for p in group['ipts_params']]).item() / len(group['ipts_params'])
                for p in group['ipts_params']:
                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        # Exponential moving average importance score of ffn
                        state['exp_avg_ffn_ipt'] = torch.zeros_like(p.data)
                    exp_avg_ffn_ipt = state['exp_avg_ffn_ipt']
                    exp_avg_ffn_ipt.mul_(beta3).add_(p.grad, alpha=1.0 - beta3)
                    ffn_ipts.append(p.grad)
                    exp_avg_ffn_ipts.append(exp_avg_ffn_ipt)

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_inf'] = torch.zeros_like(p.data)

                exp_avg, exp_inf = state['exp_avg'], state['exp_inf']
                eps = group['eps']
                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # Update the exponentially weighted infinity norm.
                norm_buf = torch.cat([
                    exp_inf.mul_(beta2).unsqueeze(0),
                    grad.abs().add_(eps).unsqueeze_(0)
                ], 0)
                torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))
                update = exp_avg / (exp_inf + eps)

                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data


                if group['params_type'] == "head":
                    if group["weight_decay"] > 0.0:
                        layer_id = i//4
                    else:
                        layer_id = i//3

                    exp_avg_head_ipt = exp_avg_head_ipts[layer_id]
                    head_ipt = head_ipts[layer_id]

                    if update_record and group["weight_decay"] > 0.0:
                        if (layer_id, "head") not in actual_lr:
                            actual_lr[(layer_id, "head")] = lr_scheduled

                    if group["weight_decay"] > 0.0 and i%4 == 3:
                        lr_scheduled = lr_scheduled * torch.ones_like(p.data).view(p.data.shape[0], head_ipt.shape[0], -1) \
                                                    * ((head_ipt - exp_avg_head_ipt).abs() / (head_ipt + eps**2))[None, :, None]
                    else:
                        lr_scheduled = lr_scheduled * torch.ones_like(p.data).view(head_ipt.shape[0], -1) \
                                                    * ((head_ipt - exp_avg_head_ipt).abs() / (head_ipt + eps**2))[:, None]
                    p.data -= lr_scheduled.view(p.data.shape) * update
                    # update_with_lr = lr_scheduled * update
                    # p.data.add_(-update_with_lr)

                    if update_record and group["weight_decay"] > 0.0:
                        if (layer_id, "head") not in modulated_lr:
                            modulated_lr[(layer_id, "head")] = lr_scheduled.squeeze().clone()
                        if (layer_id, "head") not in ipts:
                            ipts[(layer_id, "head")] = head_ipt.squeeze().clone()
                        if (layer_id, "head") not in exp_avg_ipts:
                            exp_avg_ipts[(layer_id, "head")] = exp_avg_head_ipt.squeeze().clone()

                elif group['params_type'] == "ffn":
                    layer_id = i//2
                    exp_avg_ffn_ipt = exp_avg_ffn_ipts[layer_id]
                    ffn_ipt = ffn_ipts[layer_id]

                    if update_record and group["weight_decay"] > 0.0:
                        if (layer_id, "ffn") not in actual_lr:
                            actual_lr[(layer_id, "ffn")] = lr_scheduled

                    lr_scheduled = lr_scheduled * ((ffn_ipt - exp_avg_ffn_ipt).abs() / (ffn_ipt + eps**2)).item()
                    update_with_lr = lr_scheduled * update
                    p.data.add_(-update_with_lr)

                    if update_record and group["weight_decay"] > 0.0:
                        if (layer_id, "ffn") not in modulated_lr:
                            modulated_lr[(layer_id, "ffn")] = lr_scheduled
                        if (layer_id, "ffn") not in ipts:
                            ipts[(layer_id, "ffn")] = ffn_ipt.clone()
                        if (layer_id, "ffn") not in exp_avg_ipts:
                            exp_avg_ipts[(layer_id, "ffn")] = exp_avg_ffn_ipt.clone()

                else:
                    update_with_lr = lr_scheduled * update
                    p.data.add_(-update_with_lr)

                state['step'] += 1

        if update_record:
            self.update_sensitivity_record(actual_lr, modulated_lr, ipts, exp_avg_ipts)

        return loss

class UnstructAwareAdamax(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    by xiaodl
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 betas=(0.9, 0.999), beta3=0.99, eps=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        betas = betas + (beta3,)
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[2]))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        self.init_sensitivity_record()
        super(UnstructAwareAdamax, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_inf'].to(device)
            state['exp_avg_ipt'].to(device)

    def init_sensitivity_record(self):
        self.modulated_lrs = {}
        self.ipt_vars = {}
        self.ipts = {}

    def update_sensitivity_record(self, key, modulated_lr=None, ipt_var=None, ipt=None):

        if key not in self.modulated_lrs:
            self.modulated_lrs[key] = []
            self.ipt_vars[key] = []
            self.ipts[key] = []

        if modulated_lr is not None:
            self.modulated_lrs[key].append(modulated_lr)
        if ipt_var is not None:
            self.ipt_vars[key].append(ipt_var)
        if ipt is not None:
            self.ipts[key].append(ipt)

    def get_sensitivity_record(self):
        return self.modulated_lrs, self.ipt_vars, self.ipts

    def step(self, closure=None, update_record=False):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]

            if group['params_type'] == 'common':
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_inf'] = torch.zeros_like(p.data)

                    exp_avg, exp_inf = state['exp_avg'], state['exp_inf']

                    eps = group['eps']
                    # Add grad clipping
                    if group['max_grad_norm'] > 0:
                        clip_grad_norm_(p, group['max_grad_norm'])

                    # Update biased first moment estimate.
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    # Update the exponentially weighted infinity norm.
                    norm_buf = torch.cat([
                        exp_inf.mul_(beta2).unsqueeze(0),
                        grad.abs().add_(eps).unsqueeze_(0)
                    ], 0)
                    torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))
                    update = exp_avg / (exp_inf + eps)

                    if group['weight_decay'] > 0.0:
                        update += group['weight_decay'] * p.data

                    if group['t_total'] != -1:
                        schedule_fct = schedule_func(group['schedule'])
                        lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                    else:
                        lr_scheduled = group['lr']

                    update_with_lr = lr_scheduled * update
                    p.data.add_(-update_with_lr)
                    state['step'] += 1
            else:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_inf'] = torch.zeros_like(p.data)
                        state['exp_avg_ipt'] = torch.zeros_like(p.data)

                    exp_avg, exp_inf, exp_avg_ipt = state['exp_avg'], state['exp_inf'], state['exp_avg_ipt']
                    eps = group['eps']
                    # Add grad clipping
                    if group['max_grad_norm'] > 0:
                        clip_grad_norm_(p, group['max_grad_norm'])

                    # Update biased first moment estimate.
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    # Update the exponentially weighted infinity norm.
                    norm_buf = torch.cat([
                        exp_inf.mul_(beta2).unsqueeze(0),
                        grad.abs().add_(eps).unsqueeze_(0)
                    ], 0)
                    torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))
                    update = exp_avg / (exp_inf + eps)
                    # Update moving average of importance score
                    ipt = (p.data * grad).abs()

                    exp_avg_ipt.mul_(beta3).add_(ipt, alpha=1.0 - beta3)

                    if group['t_total'] != -1:
                        schedule_fct = schedule_func(group['schedule'])
                        lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                    else:
                        lr_scheduled = group['lr']

                    if group['weight_decay'] > 0.0:
                        update += group['weight_decay'] * p.data

                    lr_scheduled *= (ipt - exp_avg_ipt).abs() / (exp_avg_ipt + eps**2)
                    ###################################
                    if update_record:
                        if group['weight_decay'] > 0:
                            self.update_sensitivity_record((group['weight_decay'], i), modulated_lr=lr_scheduled)
                    ###################################

                    update_with_lr = lr_scheduled * update
                    p.data.add_(-update_with_lr)
                    state['step'] += 1

        return loss
