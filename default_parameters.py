#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :equation.py
# @Time      :2024/3/12 16:22
# @Author    :Feiyu
# @Main      :Configurations for equation construction and training process.

import os
import torch
import numpy as np


default_Config = {
    'device': 'cpu',
    'n_layer' : 4,
    'batch_size' : 256,
    'valid_size' : 64,
    'logging_frequency':1,
    'learning_rate':0.001,
    'weight_decay':0,
    'early_stop':True,
    'num_iterations':20000,
    'stop_patience':1000,
    'verbose':1,
    'TH_DTYPE':torch.float32,
    'DELTA_CLIP':50,
}

AllenCahnConfig = {
    'name' : "Allen_Cahn_Equation",
    'model_save_dir': os.path.join("saved_models", "Allen_Cahn_Equation"),
    'total_time' : 0.3,
    'num_time_interval' : 20,
    'dim' : 100,
    'default_Config': default_Config,
    'num_hiddens' : [100, 110, 100, 100],
    'y_init_range' : [0.3, 0.6],
}

HJBConfig = {
    'name': "Hamilton_Jacobi_Bellman_Equation",
    'model_save_dir': os.path.join("saved_models", "Hamilton_Jacobi_Bellman_Equation"),
    # Y_0 is about 4.5901.
    'dim' : 100,
    'total_time' : 1.0,
    'num_time_interval' : 20,
    'default_Config': default_Config,
    'num_hiddens' : [100, 110, 110, 100],
    'y_init_range' : [0, 1],
}

PricingOptionConfig = {
    'name': "Nonlinear_Black_Scholes_European_Diff_Rates",
    'model_save_dir': os.path.join("saved_models", "Nonlinear_Black_Scholes_European_Diff_Rates"),
    'dim' : 100,
    'total_time' : 0.5,
    'num_time_interval' : 20,
    'default_Config': default_Config,
    'num_hiddens' : [100, 110, 100, 100],
    'y_init_range' : [15, 18],
}

PricingDefaultRiskConfig = {
    'name': "Nonlinear_Black_Scholes_default_risk",
    'model_save_dir': os.path.join("saved_models", "Nonlinear_Black_Scholes_default_risk"),
    'dim' : 100,
    'total_time' : 1,
    'num_time_interval' : 40,
    'default_Config': default_Config,
    'num_hiddens' : [100, 110, 100, 100],
    'y_init_range' : [40, 50],
}

BurgesTypeConfig = {
    'name': "Burgers_Type_Equation",
    'model_save_dir': os.path.join("saved_models", "Burgers_Type_Equation"),
    'dim' : 50,
    'total_time' : 0.2,
    'num_time_interval' : 30,
    'default_Config': default_Config,
    'num_hiddens' : [50, 60, 60, 50],
    'y_init_range' : [2, 4],

}

QuadraticGradientsConfig = {
    'name': "PDE_Quadratically_Growing_Derivatives",
    'model_save_dir': os.path.join("saved_models", "PDE_Quadratically_Growing_Derivatives"),
    'dim' : 100,
    'total_time' : 1.0,
    'num_time_interval' : 30,
    'default_Config': default_Config,
    'num_hiddens' : [100, 110, 100, 100],
    'y_init_range' : [2, 4],
}


ReactionDiffusionConfig = {
    'name': "Time_Dependent_Reaction_Diffusion_Equation",
    'model_save_dir': os.path.join("saved_models", "Time_Dependent_Reaction_Diffusion_Equation"),
    'dim' : 100,
    'total_time' : 1.0,
    'num_time_interval' : 30,
    'default_Config': default_Config,
    'num_hiddens' : [100, 110, 100, 100],
    'y_init_range': [0.3, 0.6],
}

