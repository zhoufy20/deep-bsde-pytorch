#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :main.py
# @Time      :2024/3/13 0:17
# @Author    :Feiyu
# @Main      ：

if __name__ == "__main__":
    import equation
    import default_parameters
    from train import BSDESolver
    from lib import draw_dw_x

    # # # AllenCahn
    AllenCahn = equation.AllenCahn(default_parameters.AllenCahnConfig)
    model = BSDESolver(default_parameters.AllenCahnConfig, AllenCahn)
    # draw_dw_x(100, default_parameters.AllenCahnConfig, equation.AllenCahn)
    model.solve()
    #
    #
    # # # HJBLQ
    # HJBLQ = equation.HJBLQ(default_parameters.HJBConfig)
    # model = BSDESolver(default_parameters.HJBConfig, HJBLQ)
    # draw_dw_x(100, default_parameters.HJBConfig, equation.HJBLQ)
    # model.solve()

    # # PricingDefaultRisk
    # PricingDefaultRisk = equation.PricingDefaultRisk(default_parameters.PricingDefaultRiskConfig)
    # model =BSDESolver(default_parameters.PricingDefaultRiskConfig, PricingDefaultRisk)
    # draw_dw_x(100, default_parameters.PricingDefaultRiskConfig, equation.PricingDefaultRisk)
    # model.solve()
    #
    # # PricingDiffRate
    # PricingDiffRate = equation.PricingDiffRate(default_parameters.PricingOptionConfig)
    # model =BSDESolver(default_parameters.PricingOptionConfig, PricingDiffRate)
    # draw_dw_x(100, default_parameters.PricingOptionConfig, equation.PricingDiffRate)
    # model.solve()
    #
    # Burgers = equation.BurgersType(default_parameters.BurgesTypeConfig)
    # model = BSDESolver(default_parameters.BurgesTypeConfig, Burgers)
    # draw_dw_x(100, default_parameters.BurgesTypeConfig, equation.BurgersType)
    # model.solve()
    #
    # # PDE_Quadratically_Growing_Derivatives
    # QuadraticGradient = equation.QuadraticGradient(default_parameters.QuadraticGradientsConfig)
    # model = BSDESolver(default_parameters.QuadraticGradientsConfig, QuadraticGradient)
    # # draw_dw_x(100, default_parameters.BurgesTypeConfig, equation.BurgersType)
    # model.solve()
    #
    # # # Time_Dependent_Reaction_Diffusion_Equation
    # ReactionDiffusion = equation.ReactionDiffusion(default_parameters.ReactionDiffusionConfig)
    # model = BSDESolver(default_parameters.ReactionDiffusionConfig, ReactionDiffusion)
    # # draw_dw_x(100, default_parameters.BurgesTypeConfig, equation.BurgersType)
    # model.solve()
