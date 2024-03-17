#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :equation.py
# @Time      :2024/3/12 16:22
# @Author    :Feiyu
# @Main      ：the main framwork of DNN

import os
import time
import torch
import numpy as np
from lib import EarlyStopping, save_model, save_state_dict, load_state_dict

### the paramete of BatchNorm1d
MOMENTUM = 0.99
EPSILON = 1e-6


class Dense(torch.nn.Module):
    def __init__(self, cin, cout, batch_norm=True, activate=True):
        super(Dense, self).__init__()
        self.cout = cout
        self.linear = torch.nn.Linear(cin, cout)
        self.activate = activate
        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(cout,eps=EPSILON, momentum=MOMENTUM)
        else:
            self.bn = None
        torch.nn.init.normal_(self.linear.weight,std=5.0/np.sqrt(cin+cout))

    def forward(self,x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activate:
            x = torch.nn.functional.relu(x)
        return x


class Subnetwork(torch.nn.Module):
    def __init__(self, config):
        super(Subnetwork, self).__init__()
        self._config = config
        self.bn = torch.nn.BatchNorm1d(config['dim'],eps=EPSILON, momentum=MOMENTUM)
        self.layers = [Dense(config['num_hiddens'][i-1], config['num_hiddens'][i]) for i in range(1, len(config['num_hiddens'])-1)]
        self.layers += [Dense(config['num_hiddens'][-2], config['num_hiddens'][-1], activate=False)]
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self,x):
        x = self.bn(x)
        x = self.layers(x)
        return x


class ForwardModel(torch.nn.Module):
    def __init__(self, config, bsde):
        super(ForwardModel, self).__init__()
        self.config = config
        self.bsde = bsde
        self.device = config['default_Config']['device']
        self.TH_DTYPE = config['default_Config']['TH_DTYPE']
        self.DELTA_CLIP = config['default_Config']['DELTA_CLIP']

        # make sure consistent with FBSDE equation
        self.dim = bsde.eqn_dim

        self.num_time_interval = bsde.eqn_num_time_interval
        self.total_time = bsde.eqn_total_time
        self.y_init = torch.nn.Parameter(torch.Tensor([1]))
        self.y_init.data.uniform_(self.config['y_init_range'][0], self.config['y_init_range'][1])
        self.subnetworkList = torch.nn.ModuleList([Subnetwork(config) for _ in range(self.num_time_interval-1)])

    def forward(self, x, dw):
        time_stamp = np.arange(0, self.bsde.eqn_num_time_interval) * self.bsde.eqn_delta_t
        z_init = (torch.zeros([1, self.dim]).uniform_(-0.1, 0.1).to(self.TH_DTYPE)).to(self.device)
        # dw' size=[num_sample, self.eqn_dim, self.eqn_num_time_interval]
        # all_one_vec' size=[num_sample, 1]
        # all_one_vec = torch.ones((dw.shape[0], 1), dtype=self.TH_DTYPE).to(self.device)
        # y' size=[num_sample, 1]
        all_one_vec = torch.ones((dw.shape[0], 1), dtype=self.TH_DTYPE).to(self.device)

        y = all_one_vec * self.y_init
        z = torch.matmul(all_one_vec, z_init)
        # z'size=[num_sample, self.dim]

        for t in range(0, self.num_time_interval-1):
            y = y - self.bsde.eqn_delta_t * (self.bsde.f_th(time_stamp[t], x[:, :, t], y, z))
            # dw[:, :, t].size=[num_sample, self.dim]
            add = torch.sum(z * dw[:, :, t], dim=1, keepdim=True)
            y = y + add
            z = self.subnetworkList[t](x[:, :, t + 1]) / self.dim

        # # terminal time
        y = y - self.bsde.eqn_delta_t * self.bsde.f_th(time_stamp[-1], x[:, :, -2], y, z) + torch.sum(z * dw[:, :, -1], dim=1, keepdim=True)

        # use linear approximation outside the clipped range
        # delta = y - self.bsde.g_th(self.total_time, x[:, :, -1])
        # loss = torch.mean(torch.where(torch.abs(delta) < self.DELTA_CLIP, delta ** 2,
        #                               2 * self.DELTA_CLIP * torch.abs(delta) - self.DELTA_CLIP ** 2))

        # torch.nn.MSELoss()
        criterion = torch.nn.MSELoss()
        loss = criterion(y, self.bsde.g_th(self.bsde.eqn_total_time, x[:, :, -1]))
        return loss, self.y_init

class BSDESolver(torch.nn.Module):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        super(BSDESolver, self).__init__()
        self.config = config
        self.bsde = bsde
        self.device = config['default_Config']['device']
        self.TH_DTYPE = config['default_Config']['TH_DTYPE']
        self.verbose = config['default_Config']['verbose']
        self.DELTA_CLIP = config['default_Config']['DELTA_CLIP']

        # prepare out file
        if not os.path.exists(config['model_save_dir']):
            os.makedirs(config['model_save_dir'], exist_ok=True)
        self.log = open(os.path.join(config['model_save_dir'], 'train.log'),
                        'w', buffering=1)

        # check device
        if torch.cuda.is_available() and self.device == 'cpu':
            print('User warning: `CUDA` device is available, but you choosed `cpu`.', file=self.log)
        elif not torch.cuda.is_available() and self.device.split(':')[0] == 'cuda':
            print('User warning: `CUDA` device is not available, but you choosed `cuda:0`. '
                  'Change the device to `cpu`.', file=self.log)
            self.device = 'cpu'
        print('User info: Specified device for potential models:', self.device, file=self.log)

    def solve(self):
        start_time = time.time()
        dw_valid, x_valid = self.bsde.sample(self.config['default_Config']['valid_size'])
        dw_test, x_test = self.bsde.sample(self.config['default_Config']['valid_size'])

        # construct a models and an optimizer.
        model = ForwardModel(self.config, self.bsde).to(self.device)


        # select the optimizer from SGD or Adam or AdamW
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.config['default_Config']['learning_rate'],
        #                              weight_decay=self.config['default_Config']['weight_decay'])

        # optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['default_Config']['learning_rate'],
        #                              weight_decay=self.config['default_Config']['weight_decay'])

        optimizer = torch.optim.SGD(model.parameters(), lr=self.config['default_Config']['learning_rate'],
                                     weight_decay=self.config['default_Config']['weight_decay'])

        # Loading state dict (models weigths and other info) from the disk.
        if os.path.exists(os.path.join(self.config['model_save_dir'],'pdes.pth')):
            try:
                checkpoint = load_state_dict(self.config['model_save_dir'])
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                model = model.to(self.device)
                model.device=self.device
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f'User info: Model and optimizer state dict loaded successfully \n from {self.config['model_save_dir']}.', file=self.log)
            except:
                print('User warning: Exception catched when loading models and optimizer state dict.', file=self.log)
        else:
            print('User info: Checkpoint not detected', file=self.log)

        # early stop
        if self.config['default_Config']['early_stop']:
            stopper = EarlyStopping(model, self.log,
                                    patience=self.config['default_Config']['stop_patience'],
                                    model_save_dir=self.config['model_save_dir'])

        # log file
        print('========================================================================', file=self.log)
        print(model, file=self.log)
        print('========================================================================', file=self.log)
        print("{:0>5s}   {:>16s}   {:>16s}   {:>16s}".format( "Epoch", "Loss", "target_value", "elapsed_Time"),
              file=self.log)

        # start the training
        loss_train_all, y_init_train_all = [], []
        loss_valid_all, y_init_valid_all = [], []
        for step in range(self.config['default_Config']['num_iterations'] + 1):
            dw_train, x_train = self.bsde.sample(self.config['default_Config']['batch_size'])
            optimizer.zero_grad()
            model.train()
            loss_train, init_train = model(x_train, dw_train)
            loss_train.backward()
#            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.DELTA_CLIP, norm_type=2)
            optimizer.step()

            elapsed_time = time.time() - start_time
            loss_train_all.append(loss_train.item())
            y_init_train_all.append(init_train.item())
            if self.verbose > 1:
                if step % self.config['default_Config']['logging_frequency'] == 0:
                    print("{:0>5d}   {:>16.14f}   {:>16.14f}   {:>10.1f}   Train_info".format(
                        step, loss_train, init_train.item(), elapsed_time), file=self.log)


            # validation every epoch
            with torch.no_grad():
                model.eval()
                loss_valid, init_valid = model(x_valid, dw_valid)
                loss_valid_all.append(loss_valid.item())
                y_init_valid_all.append(init_valid.item())
                if self.verbose > 0:
                    if step % self.config['default_Config']['logging_frequency'] == 0:
                        print("{:0>5d}   {:>16.14f}   {:>16.14f}   {:>10.1f}   Validation_info".format(
                            step, loss_valid.item(), init_valid.item(), elapsed_time), file=self.log)

            # loss_valid_all = loss_valid_all.cpu().numpy()
            # y_init_valid_all = y_init_valid_all.cpu().numpy()
            if self.config['default_Config']['early_stop']:
                if stopper.step(loss_valid, step, model, optimizer):
                    break
                if stopper.update:
                    np.savetxt(os.path.join(self.config['model_save_dir'],'loss.txt'),
                               loss_valid_all, fmt='%.8f')
                    np.savetxt(os.path.join(self.config['model_save_dir'], 'init.txt'),
                               y_init_valid_all, fmt='%.8f')
            else:
                save_model(model, model_save_dir=self.config['model_save_dir'])
                save_state_dict(model, state_dict_save_dir=self.config['model_save_dir'])


        # test every epoch
        with torch.no_grad():

            model.eval()
            loss_test, init_test = model(dw_test, x_test)

            print(f'''User info, models performance on testset: (No sample weight on the loss)
                    Epoch      : {step}
                    loss:      : {loss_test.item()}
                    init       : {init_test.item()}
                    Dur (s)    : {elapsed_time}''', file=self.log)

        self.log.close()






if __name__ == '__main__':
    # from equation import AllenCahn
    # from default_parameters import AllenCahnConfig
    # allen_cahn = AllenCahn(AllenCahnConfig)
    # # print(allen_cahn.eqn_num_time_interval)
    # model = BSDESolver(AllenCahnConfig,allen_cahn)
    # model.solve()

    from equation import HJBLQ
    from default_parameters import HJBConfig
    HJB = HJBLQ(HJBConfig)
    # print(allen_cahn.eqn_num_time_interval)
    model = BSDESolver(HJBConfig, HJB)
    model.solve()


