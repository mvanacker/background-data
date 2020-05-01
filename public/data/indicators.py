import pandas as pd
import numpy as np
from dateutil.parser import isoparse

# General smoothing function
def smooth_series(series, period=3, iters=3):
  res = series
  name = f'ema-{period}'
  for _ in range(iters):
    f = ema_series(res, period)
    g = ema(f, period, name)
    res = f[name] * 1.5 - g[name] * .5
  return res

# Auxiliary function for exponentially weighted averages
def prep_for_ewm(series, period):
  sma = series.rolling(window=period, min_periods=period).mean()[:period]
  return pd.concat([sma, series[period:]])

# Exponential moving average
def ema(df, period, src='close'):
  return ema_series(df[src], period)
def ema_series(series, period):
  ema = prep_for_ewm(series, period).ewm(span=period, adjust=False).mean()
  return pd.DataFrame({f'ema-{period}': ema})

# RSI moving average
def rma(df, period, src='close'):
  return rma_series(df[src], period)
def rma_series(series, period):
  rma = prep_for_ewm(series, period).ewm(com=period - 1, adjust=False).mean()
  return pd.DataFrame({'rsi': rma})

# Stochastic
def stoch(df, src='close', low='low', high='high',
              periodK=14, periodD=3, smoothK=6):
  return stoch_series(df[src], df[low], df[high], periodK, periodD, smoothK)
def stoch_series(src, low, high, periodK=14, periodD=3, smoothK=6):
  k = stoch_simple(src, low, high, periodK).rolling(window=smoothK).mean()
  d = k.rolling(window=periodD).mean()
  return pd.DataFrame({f'K': k, 'D': d})
def stoch_simple(src, low, high, period=14):
  lows = low.rolling(window=period).min()
  highs = high.rolling(window=period).max()
  return ((src - lows) / (highs - lows)) * 100

# Smooth Stoch
def smooth_stoch(df, src='close', low='low', high='high',
                     period=14, periodSmooth=3, itersSmooth=3, periodSignal=3):
  return smooth_stoch_series(df[src], df[low], df[high], period,
                             periodSmooth, itersSmooth, periodSignal)
def smooth_stoch_series(series, lows, highs, period=14, periodSmooth=3, itersSmooth=3, periodSignal=3):
  simple = stoch_simple(series, lows, highs, period)
  smooth_stoch = smooth_series(simple, periodSmooth, itersSmooth)
  smooth_stoch_df = pd.DataFrame({'K': smooth_stoch})
  smooth_stoch_df['D'] = ema_series(smooth_stoch, periodSignal)[f'ema-{periodSignal}']
  return smooth_stoch_df

# Relative Strength Index
def rsi(df, period=14, src='close'):
  change = df[src].diff()
  up = rma_series(change.clip(lower=0), period)
  down = rma_series(-change.clip(upper=0), period)
  return (100 - (100 / (1 + up / down)))

# RSI + EMA
def rsi_ema(df, period=14, ema_period=7, src='close'):
  _rsi = rsi(df, period, src)
  _rsi['rsi-ema'] = ema(_rsi, ema_period, 'rsi')
  return _rsi

# Historic Volatility Percentile
def hvp(df, src='close', hv_period=10, per_period=100, ma_len=20):
  return hvp_series(df[src], hv_period, per_period, ma_len)
def hvp_series(series, hv_period=10, per_period=100, ma_len=20):
  hv = np.log(series).diff().rolling(window=hv_period).std()
  percentile = lambda xs: 100 * sum(xs < xs[-1]) / per_period
  hvp_series = hv.rolling(window=per_period + 1).apply(percentile, raw=False)
  hvp_ma = hvp_series.rolling(window=ma_len).mean()
  return pd.DataFrame({'hvp': hvp_series, 'hvp-ma': hvp_ma})

# Inefficient but general approach to computing partial indicators
def part_ind(df, partial, indicator, slice_needed=14):
  partial = {k: [d] for k, d in partial.items()}
  partial_row = pd.DataFrame.from_dict(partial)
  partial_row['timestamp'] = partial_row['timestamp'].apply(isoparse)
  partial_row = partial_row.set_index('timestamp')
  return indicator(df.append(partial_row, sort=False))[-1:]