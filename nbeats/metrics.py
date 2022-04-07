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
Metrics functions using numpy arrays.
"""
import numpy as torch
import torch

Forecast = torch.ndarray
Target = torch.ndarray


def mase(forecast, insample, outsample, frequency):
    """
    MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

    :param forecast: Forecast values. Shape: batch, time_o
    :param insample: Insample values. Shape: batch, time_i
    :param outsample: Target values. Shape: batch, time_o
    :param frequency: Frequency value
    :return: Same shape array with error calculated for each time step
    """
    return torch.mean(torch.abs(forecast - outsample)) / torch.mean(torch.abs(insample[:-frequency] - insample[frequency:]))


def nd(forecast, target):
    """
    Normalized deviation as defined in https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf

    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :return: Error value
    """
    return torch.mean(torch.abs(target - forecast)) / torch.mean(torch.abs(target))


def nrmse(forecast, target):
    """
    Normalized RMSE as defined in https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf

    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :return: Error values
    """
    return torch.sqrt(torch.mean(torch.power((forecast - target), 2))) / (torch.mean(torch.abs(target)))


def mape(forecast, target):
    """
    MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    :param forecast: Predicted values.
    :param target: Target values.
    :return: Same shape array with error calculated for each time step
    """
    return 100 * torch.abs(forecast - target) / target


def smape_1(forecast, target):
    """
    sMAPE loss as defined in "Appendix A" of
    http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf

    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :return: Same shape array with error calculated for each time step
    """
    return 200 * torch.abs(forecast - target) / (target + forecast)


def smape_2(forecast, target):
    """
    sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :return: Same shape array with sMAPE calculated for each time step of each timeseries.
    """
    denom = torch.abs(target) + torch.abs(forecast)
    # divide by 1.0 instead of 0.0, in case when denom is zero the enumerator will be 0.0 anyway.
    denom[denom == 0.0] = 1.0
    return 200 * torch.abs(forecast - target) / denom