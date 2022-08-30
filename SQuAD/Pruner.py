import os
import sys 
import argparse
import logging
import random
import torch
import numpy as np


class Pruner(object):
    def __init__(self, model, args, total_step, tb_writer=None, \
                mask_param_name=['attention.self', 'attention.output.dense',\
                'output.dense', 'intermediate.dense'], non_mask_name = ["embedding", "norm"], \
                use_no_mask=False, pruner_name='PLATON' ):
        self.model = model
        self.config = vars(args)
        self.args = args
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.mask_param_name = mask_param_name 
        self.non_mask_name = non_mask_name 
        self.use_no_mask = use_no_mask
        self.total_step = total_step
        self.tb_writer = tb_writer
        self.pruner_name = pruner_name
        self.beta1 = self.config["beta1"]
        self.beta2 = self.config["beta2"]
        self.deltaT = self.config["deltaT"]


    def whether_mask_para(self, n):
        if not self.use_no_mask:
            return any(nd in n for nd in self.mask_param_name)
        else:
            return not any([nd in n for nd in self.non_mask_name])


    def schedule_threshold_comb(self, step: int):
        # Schedule the ramining ratio 
        args = self.args
        total_step = self.total_step
        initial_threshold = self.config['initial_threshold']
        final_threshold = self.config['final_threshold']
        initial_warmup = self.config['initial_warmup']
        final_warmup = self.config['final_warmup']
        warmup_steps = self.config['warmup_steps']
        mask_ind = False
        if step <= initial_warmup * warmup_steps:
            threshold = initial_threshold
            mask_ind = False
        elif step > (total_step - final_warmup * warmup_steps):
            threshold = final_threshold
            mask_ind = True
        else: 
            spars_warmup_steps = initial_warmup*warmup_steps
            spars_schedu_steps = (final_warmup+initial_warmup)*warmup_steps
            mul_coeff = 1-(step-spars_warmup_steps)/(total_step-spars_schedu_steps)
            threshold = final_threshold+(initial_threshold-final_threshold)*(mul_coeff** 3)
            mask_ind = True if step % self.deltaT == 0 else False
        return threshold, mask_ind


    def update_ipt_with_local_window(self, model, global_step):
        # Calculate the sensitivity and uncertainty 
        for n,p in model.named_parameters():
            if self.whether_mask_para(n):
                if n not in self.exp_avg_ipt:
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.ipt[n] = torch.zeros_like(p)
                    if self.beta2>0 and self.beta2!=1:
                        self.exp_avg_unc[n] = torch.zeros_like(p)
                if self.pruner_name == 'Magnitude':
                    # Calculate the score of magnitude pruning
                    self.ipt[n] = p.abs().detach()
                elif self.pruner_name == 'PLATON':
                    local_step = global_step % self.deltaT
                    update_step = global_step // self.deltaT
                    if local_step == 0: 
                        self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                        if self.beta2 > 0 and self.beta2 < 1:
                            self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                                  (1 - self.beta2) * (self.ipt[n]-self.exp_avg_ipt[n]).abs()
                        elif self.beta2 == 2.:
                            self.exp_avg_unc[n] = (update_step*self.exp_avg_unc[n] + \
                                                    (self.ipt[n]-self.exp_avg_ipt[n])**2 )/(update_step+1)
                        self.ipt[n] = (p * p.grad).abs().detach()
                    else:
                        self.ipt[n] = (self.ipt[n]*local_step+(p*p.grad).abs().detach())/(local_step+1)
                else:
                    raise ValueError("Incorrect Pruner Name.")


    def mask_with_threshold(self, model, threshold):
        # Calculate the final importance score
        is_dict = {}
        for n,p in model.named_parameters():
            if self.whether_mask_para(n):
                if self.pruner_name == 'Magnitude':
                    is_dict[n] = self.ipt[n]
                elif self.pruner_name == 'PLATON':
                    if self.beta2 > 0 and self.beta2<1:
                        is_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc[n] 
                    elif self.beta2 == 1.:
                        is_dict[n] = self.exp_avg_ipt[n]
                    elif self.beta2 == 2.:
                        is_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc.sqrt()
                    else:
                        # Handling the uncepted beta2 to default setting 
                        is_dict[n] = self.exp_avg_ipt[n] * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                else:
                    raise ValueError("Incorrect Pruner Name.")
        # Calculate the mask threshold 
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
        mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0]*(1 - threshold)))[0].item()
        # Mask weights whose importance lower than threshold 
        for n,p in model.named_parameters():
            if self.whether_mask_para(n):
                p.data.masked_fill_(is_dict[n] < mask_threshold, 0.0)
        return mask_threshold


    def update_and_pruning(self, model, global_step):
        # Update importance score after optimizer stepping
        self.update_ipt_with_local_window(model, global_step)
        # Get the ramaining ratio 
        threshold, mask_ind = self.schedule_threshold_comb(global_step)
        if mask_ind:
            # Mask weights during masking horizon 
            mask_threshold = self.mask_with_threshold(model, threshold)
        else:
            mask_threshold = None
        return threshold, mask_threshold
