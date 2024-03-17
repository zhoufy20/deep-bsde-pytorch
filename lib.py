#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :lib.py
# @Time      :2024/3/13 16:04
# @Author    :Feiyu
# @Main      ：some libraries such as EarlyStopping, saving models.


import os
import torch
import matplotlib.pyplot as plt
import default_parameters
import equation


class EarlyStopping:
    def __init__(self, model, logger, patience=10, model_save_dir='model_save_dir'):
        """Stop training when models performance stop improving after some steps."""
        self.model      = model
        self.patience   = patience
        self.counter    = 0
        self.best_score = None
        self.update     = None
        self.early_stop = False
        self.logger     = logger
        self.model_save_dir = model_save_dir

        if not os.path.exists(self.model_save_dir):
            os.mkdir(model_save_dir)

    def step(self, score, epoch, model, optimizer):
        if self.best_score is None:
            self.best_score = score
            self.update = True
            # self.save_model(models, model_save_dir=self.model_save_dir)
            # self.save_checkpoint(models, model_save_dir=self.model_save_dir)
        elif score > self.best_score:
            self.update = False
            self.counter += 1
            print(f'User log: EarlyStopping counter: {self.counter} out of {self.patience}',
                  file=self.logger)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.update     = True
            self.save_model(model)
            save_state_dict(model, state_dict_save_dir=self.model_save_dir,
                            optimizer_state_dict=optimizer.state_dict(),
                            epoch=epoch, total_loss=score)
            self.counter = 0

        return self.early_stop

    def save_model(self, model):
        '''Saves models when validation loss decrease.'''
        torch.save(model, os.path.join(self.model_save_dir, 'pdes.pth'))
        print(f'User info: Save models with the best score: {self.best_score}',
              file=self.logger)


def save_model(model, model_save_dir='agat_model'):
    """Saving PyTorch models to the disk."""
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(model, os.path.join(model_save_dir, 'pdes.pth'))

def load_model(model_save_dir='agat_model', device='cuda'):
    """Loading PyTorch models from the disk."""
    device = torch.device(device)
    if device.type == 'cuda':
        new_model = torch.load(os.path.join(model_save_dir, 'pdes.pth'))
    elif device.type == 'cpu':
        new_model = torch.load(os.path.join(model_save_dir, 'pdes.pth'),
                               map_location=torch.device(device))
    new_model.eval()
    new_model = new_model.to(device)
    new_model.device = device
    return new_model

def save_state_dict(model, state_dict_save_dir='agat_model', **kwargs):
    """Saving state dict (models weigths and other input info) to the disk."""
    if not os.path.exists(state_dict_save_dir):
        os.makedirs(state_dict_save_dir)
    checkpoint_dict = {**{'model_state_dict': model.state_dict()}, **kwargs}
    torch.save(checkpoint_dict, os.path.join(state_dict_save_dir, 'pdes_state_dict.pth'))

def load_state_dict(state_dict_save_dir='agat_model'):
    """Loading state dict (models weigths and other info) from the disk. """

    checkpoint_dict = torch.load(os.path.join(state_dict_save_dir, 'pdes_state_dict.pth'))
    return checkpoint_dict

def draw_dw_x(num_sample, config, bsde):
    """generates simulated paths of a stochastic process X and
        the corresponding increments of Brownian motion dW"""
    pdes = bsde(config)
    dw, x = pdes.sample(num_sample)
    location = config['model_save_dir']
    plt.figure(1)
    for i in range(num_sample):
        plt.plot(x[i, 0, :])
    plt.title('the simulated paths X')
    plt.xlabel('t')
    plt.ylabel('x_sample')
    plt.savefig(os.path.join(location,'x_sample.png'))

    plt.figure(2)
    for i in range(num_sample):
        plt.plot(dw[i, 0, :])
    plt.title('the increments of Brownian motion')
    plt.xlabel('t')
    plt.ylabel('dw_sample')
    plt.savefig(os.path.join(location,'dw_sample.png'))

    plt.show()


if __name__ == "__main__":
    draw_dw_x(100, default_parameters.AllenCahnConfig, equation.AllenCahn)

