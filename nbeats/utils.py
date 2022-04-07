#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Shortcut functions to create N-BEATS models.
Created on Tue Mar  8 21:01:43 2022

"""
import numpy as np
import torch as t

from nbeats import GenericBasis, NBeats, NBeatsBlock, SeasonalityBasis, TrendBasis

def interpretable(input_size,output_size,trend_blocks,trend_layers,
                  trend_layer_size,degree_of_polynomial,
                  seasonality_blocks,seasonality_layers,
                  seasonality_layer_size,num_of_harmonics):
    """
    Create N-BEATS interpretable model.
    """
    trend_block = NBeatsBlock(input_size=input_size,
                              theta_size=2 * (degree_of_polynomial + 1),
                              basis_function=TrendBasis(degree_of_polynomial=degree_of_polynomial,
                                                        backcast_size=input_size,
                                                        forecast_size=output_size),
                              layers=trend_layers,
                              layer_size=trend_layer_size)
    seasonality_block = NBeatsBlock(input_size=input_size,
                                    theta_size=4 * int(
                                        np.ceil(num_of_harmonics / 2 * output_size) - (num_of_harmonics - 1)),
                                    basis_function=SeasonalityBasis(harmonics=num_of_harmonics,
                                                                    backcast_size=input_size,
                                                                    forecast_size=output_size),
                                    layers=seasonality_layers,
                                    layer_size=seasonality_layer_size)

    return NBeats(t.nn.ModuleList(
        [trend_block for _ in range(trend_blocks)] + [seasonality_block for _ in range(seasonality_blocks)]))

def generic(input_size, output_size, stacks, layers, layer_size):
    """
    Create N-BEATS generic model.
    """
    return NBeats(t.nn.ModuleList([NBeatsBlock(input_size=input_size,
                                               theta_size=input_size + output_size,
                                               basis_function=GenericBasis(backcast_size=input_size,
                                                                           forecast_size=output_size),
                                               layers=layers,
                                               layer_size=layer_size)
                                   for _ in range(stacks)]))
