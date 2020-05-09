import pandas as pd
import numpy as np
from math import nan, isnan

def apply_indicator(df, name, indicator, period):
  '''Generalized indicator application. Finds the relevant slice of the dataframe to apply the indicator to. Limits computations only to that slice. Finally, add the resulting series to the dataframe in the column for which the name is specified.'''
  
  # Isolate dataframe slice to which to apply indicator
  df_slice = df
  loc = 0
  if name in df:
    loc = last_valid_loc(df, name)
    df_slice = df if loc is None else df.iloc[loc - period + 2:]
  
  # Apply indicator
  values = indicator(df_slice)
  add_values(df, name, values)

  # Return the loc of the first affected row
  return len(df) if loc is None else (loc + 1 if name in df else 0)

def last_valid_loc(df, name):
  i = df[name].last_valid_index()
  return None if i is None else df.index.get_loc(i)

def add_values(df, name, values):
  '''Add the indicator's column if not yet present, otherwise only update missing.'''
  if name in df:
    df[name].update(values.dropna())
  else:
    df[name] = values

def _sma_core(period=10, src='close'):
  '''Simple moving average core functionality.'''
  return lambda df: df[src].rolling(period).mean()

def sma(df, period=10, src='close'):
  '''Add a simple moving average column to a dataframe.'''
  indicator = _sma_core(period, src)
  return apply_indicator(df, f'sma_{period}', indicator, period)

def _sma_part_core(df, part, period=10, src='close'):
  '''Compute simple moving average partial value.'''
  return (sum(df[src].iloc[1 - period:]) + part[src]) / period

def sma_part(df, part, period=10, src='close'):
  '''Compute simple moving average partial value.'''
  part[f'sma_{period}'] = _sma_part_core(df, part, period, src)

def stoch(df, period=14, src='close', high='high', low='low'):
  '''Add a (rough) stochastic column to a dataframe.'''
  def indicator(df):
    lows = df[low].rolling(period).min()
    highs = df[high].rolling(period).max()
    return 100 * (df[src] - lows) / (highs - lows)
  return apply_indicator(df, 'stoch', indicator, period)

def stoch_part(df, part, period=14, src='close', high='high', low='low'):
  '''Compute (rough) stochastic partial value.'''
  low = min(part[low], df.loc[df.index[1 - period:], low].min())
  high = max(part[high], df.loc[df.index[1 - period:], high].max())
  part['stoch'] = 100 * (part[src] - low) / (high - low)

def stoch_K(df, period=6, stoch='stoch'):
  '''Add a stochastic K column to a dataframe.'''
  indicator = _sma_core(period, src=stoch)
  return apply_indicator(df, f'stoch_K', indicator, period)

def stoch_K_part(df, part, period=6, stoch='stoch'):
  '''Compute stochastic K partial value.'''
  part['stoch_K'] = _sma_part_core(df, part, period, src=stoch)

def stoch_D(df, period=3, k='stoch_K'):
  '''Add a stochastic D column to a dataframe.'''
  indicator = _sma_core(period, src=k)
  return apply_indicator(df, f'stoch_K_D', indicator, period)

def stoch_D_part(df, part, period=3, k='stoch_K'):
  '''Compute stochastic D partial value.'''
  part['stoch_K_D'] = _sma_part_core(df, part, period, src=k)

def _ema_core(name, period=21, src='close', alpha_func=lambda p: 2 / (1 + p)):
  '''Exponential moving average core functionality. This function is not very efficient but easy to read.'''
  class Ema:
    def __init__(self, df):
      self.a = alpha_func(period)
      i = df[name].last_valid_index()
      self.last = nan if i is None else df.loc[i, name]
    def ema(self, xs):
      if len(xs) < period:
        self.last = nan
      elif isnan(self.last):
        self.last = np.mean(xs)
      else:
        self.last = self.a * xs[-1] + (1 - self.a) * self.last
      return self.last
  return lambda df: df[src].rolling(period).apply(Ema(df).ema, raw=False)

def ema(df, period=21, src='close'):
  '''Add an exponential moving average column to a dataframe.'''
  name = f'ema_{period}'
  indicator = _ema_core(name, period, src)
  return apply_indicator(df, name, indicator, period)

def ema_part(df, part, period=21, src='close'):
  '''Compute exponential moving average partial value.'''
  a = 2 / (period + 1)
  name = f'ema_{period}'
  prev = df.loc[df.index[-1], name]
  part[name] = part[src] * a + prev * (1 - a)

def change_up(df, src='close'):
  '''Add an auxiliary column for the construction of the RSI.'''
  indicator = lambda df: df[src].diff().clip(lower=0)
  return apply_indicator(df, 'change_up', indicator, period=2)

def change_part(df, part, src='close'):
  '''Compute change partial value.'''
  return part[src] - df.loc[df.index[-1], src]

def change_up_part(df, part, src='close'):
  '''Compute change up partial value.'''
  part['change_up'] = max(0, change_part(df, part, src))

def change_down(df, src='close'):
  '''Add an auxiliary column for the construction of the RSI.'''
  indicator = lambda df: -df[src].diff().clip(upper=0)
  return apply_indicator(df, 'change_down', indicator, period=2)

def change_down_part(df, part, src='close'):
  '''Compute change down partial value.'''
  part['change_down'] = -min(0, change_part(df, part, src))

def rma_change(df, name, change, period=14):
  '''Add an auxiliary column for the construction of the RSI.'''
  indicator = _ema_core(name, period, src=change, alpha_func=lambda p: 1 / p)
  return apply_indicator(df, name, indicator, period)

def rma_change_part(df, part, name, change, period=14):
  '''Compute RMA of change partial value.'''
  a = 1 / period
  last = df.loc[df.index[-1], name]
  return part[change] * a + last * (1 - a)

def rma_up(df, period=14, change_up='change_up'):
  '''Add an auxiliary column for the construction of the RSI.'''
  return rma_change(df, 'rma_up', change_up, period)

def rma_up_part(df, part, period=14, change_up='change_up'):
  '''Compute RMA up partial value.'''
  part['rma_up'] = rma_change_part(df, part, 'rma_up', change_up, period)

def rma_down(df, period=14, change_down='change_down'):
  '''Add an auxiliary column for the construction of the RSI.'''
  return rma_change(df, 'rma_down', change_down, period)

def rma_down_part(df, part, period=14, change_down='change_down'):
  '''Compute RMA down partial value.'''
  part['rma_down'] = rma_change_part(df, part, 'rma_down', change_down, period)

def rsi(df, period=14, rma_up='rma_up', rma_down='rma_down'):
  '''Add a relative strength index column to a dataframe.'''
  indicator = lambda df: 100 - 100 / (1 + df[rma_up] / df[rma_down])
  return apply_indicator(df, 'rsi', indicator, period)

def rsi_part(df, part, period=14, rma_up='rma_up', rma_down='rma_down'):
  '''Compute relative strength index partial value.'''
  part['rsi'] = 100 - 100 / (1 + part[rma_up] / part[rma_down])

def hv(df, period=10, src='close'):
  '''Add a historical volatility column to a dataframe.'''
  indicator = lambda df: np.log(df[src]).diff().rolling(period).std()
  return apply_indicator(df, 'hv', indicator, 2 + period)

def hv_part(df, part, period=10, src='close'):
  '''Compute historical volatility partial value.'''
  last = pd.Series([part[src]])
  log_src = np.log(df.loc[df.index[-period:], src].append(last))
  part['hv'] = log_src.diff().std()

def _percentile(period=100):
  '''Auxiliary percentile function.'''
  return lambda xs: 100 * sum(xs < xs[-1]) / period

def hvp(df, period=100, hv='hv'):
  '''Add a historical volatility percentile column to a dataframe.'''
  percentile = _percentile(period)
  indicator = lambda df: df[hv].rolling(period + 1).apply(percentile, raw=False)
  return apply_indicator(df, 'hvp', indicator, 1 + period)

def hvp_part(df, part, period=100, hv='hv'):
  '''Compute historical volatility percentile partial value.'''
  percentile = _percentile(period)
  last_hvs = df.loc[df.index[-period:], hv].append(pd.Series([part[hv]]))
  part['hvp'] = percentile(last_hvs)

def hvp_ma(df, period=20, hvp='hvp'):
  '''Add a historical volatility percentile moving average column to a dataframe.'''
  indicator = _sma_core(period, src=hvp)
  return apply_indicator(df, 'hvp_ma', indicator, period)

def hvp_ma_part(df, part, period=20, hvp='hvp'):
  '''Compute historical volatility percentile moving average partial value.'''
  part['hvp_ma'] = _sma_part_core(df, part, period, src=hvp)
