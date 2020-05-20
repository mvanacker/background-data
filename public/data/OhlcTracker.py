from sys import stdout
from os.path import exists
from functools import partial
from time import time, sleep, mktime
from pytz import UTC
from json import loads, dumps
from datetime import datetime, timedelta
from dateutil.parser import isoparse
from dateutil.relativedelta import relativedelta
from argparse import ArgumentParser
from requests import Session
from math import nan, isnan, sqrt
from numbers import Number
from collections import deque
from threading import Lock
from numpy import mean
import pandas as pd

from BitmexClient import BitmexClient
from Observer import Observer
from daemon import start_daemon, Starter, Stopper
from pandicators import (sma, stoch, stoch_K, stoch_D, ema, change_up,
                         change_down, rma_up, rma_down, rsi, hv, hvp, sma_part,
                         ema_part, stoch_part, stoch_K_part, stoch_D_part,
                         ema_part, change_up_part, change_down_part,
                         rma_up_part, rma_down_part, rsi_part, hv_part,
                         hvp_part, hvp_ma, hvp_ma_part)
from pricing import BlackScholes

# Setup logger
import logging
logger = logging.getLogger()
if __name__ == '__main__':
  ch = logging.StreamHandler(stdout)
  formatter = logging.Formatter("%(asctime)s: %(message)s")
  ch.setFormatter(formatter)
  logger.addHandler(ch)

# BitMEX client (ultimately only for REST API calls)
client = BitmexClient(key=None, secret=None)

# RESTful candle data API
DATA_URI = 'http://localhost:3001'
s = Session()

# Runtime constants
BINS = ['1h', '1d']
COUNT = 1000# results to fetch per request (max is 1000)
TIMEOUT = 2# seconds, to be safe
PARTIAL_DAEMON_TIMEOUT = 5# second
FORECAST_DAEMON_TIMEOUT = 300# second
LOCK = Lock()

# Resampled bins (DBINS)
to_tt = lambda t: t.timetuple()
condition_hourly = lambda t, h: t.hour % h == 0
condition_daily_mr = lambda t, m, r: mktime(t.timetuple()) % m == r
def condition_3M(t):
  tt = t.timetuple()
  return tt.tm_mday == 1 and tt.tm_mon % 3 == 1
DBINS = {
  '1h': ['2h', '3h', '4h', '6h', '12h'],
  '1d': ['2d', '3d', '1w', '1M', '3M'],
}
DCONDS = {
  '1h': {
    '2h': partial(condition_hourly, h=2),
    '3h': partial(condition_hourly, h=3),
    '4h': partial(condition_hourly, h=4),
    '6h': partial(condition_hourly, h=6),
    '12h': partial(condition_hourly, h=12),
  },
  '1d': {
    '2d': partial(condition_daily_mr, m=172800, r=169200),
    '3d': partial(condition_daily_mr, m=259200, r=82800),
    '1w': lambda t: t.timetuple().tm_wday == 0,
    '1M': lambda t: t.timetuple().tm_mday == 1,
    '3M': condition_3M,
  },
}

def resample(df, bin, dbin):
  '''Auxiliary function: resample dataframe (TODO naive).'''

  # Dataframe must be non-empty
  assert len(df)

  # Local function
  close_candle = lambda timestamp: {
    'timestamp': timestamp,
    'symbol': df['symbol'][0],
    'open': open,
    'high': high,
    'low': low,
    'close': close,
    'volume': volume,
  }

  # Initial values
  result = pd.DataFrame()
  is_opening = True
  for timestamp, candle in df.iterrows():

    # Update candle
    if is_opening:
      open = candle['open']
      high = candle['high']
      low = candle['low']
      volume = 0
      is_opening = False
    high = max(high, candle['high'])
    low = min(low, candle['low'])
    close = candle['close']
    volume += candle['volume']

    # Close candle
    if DCONDS[bin][dbin](timestamp):
      row = pd.DataFrame(close_candle(timestamp), index=[0]).set_index('timestamp')
      result = result.append(row)
      is_opening = True

  # Close condition was never met
  if result.empty:
    ft = first_timestamp(bin, dbin, df.index[0])[0]
    return result, close_candle(ft)

  partial_candle = {
    'timestamp': result.index[-1] + TIMEDELTAS[dbin],
    'symbol': df['symbol'][0],
    'open': close if is_opening else open,
    'high': close if is_opening else high,
    'low': close if is_opening else low,
    'close': close if is_opening else close,
    'volume': 0 if is_opening else volume,
  }
  return result, partial_candle

# Auxiliary function: compute next timestamp
# Note: this function is called after the condition is already met, on a
# timestamp that is already valid, so there is no need to consider leap seconds.
# Note: timestamps are in UTC so there is no need to consider daylight saving.
TIMEDELTAS = {
  '1h': timedelta(hours=1),
  '2h': timedelta(hours=2),
  '3h': timedelta(hours=3),
  '4h': timedelta(hours=4),
  '6h': timedelta(hours=6),
  '12h': timedelta(hours=12),
  '1d': timedelta(days=1),
  '2d': timedelta(days=2),
  '3d': timedelta(days=3),
  '1w': timedelta(weeks=1),
  '1M': relativedelta(months=1),
  '3M': relativedelta(months=3),
}
next_timestamp = lambda bin, t, n=1: t + n * TIMEDELTAS[bin]

# Used for resampling. They "smooth out" the closing conditions (DCONDS).
OFFSETS = {
  '1h': -1,
  '1d': 1,
}

def first_timestamp(bin, dbin, t):
  '''Find the first valid timestamp (following t) of a resampled bin.'''
  n = 0
  while not DCONDS[bin][dbin](t):
    n += 1
    t += TIMEDELTAS[bin]
  return t, n

def count(bin):
  return int(s.get(f'{DATA_URI}/candles/count?timeframe={bin}').text)

def get(bin):
  '''REST function: get candles.'''
  # uri = f'{DATA_URI}/candles?timeframe={bin}&limit=1000'
  uri = f'{DATA_URI}/candles?timeframe={bin}'
  return parseTimestamps(s.get(uri).json())

def transform(bin, candle):
  '''Transform candle before sending it off to the database.'''
  candle = dict(candle) # make (local) copy
  candle['timeframe'] = bin
  if type(candle['timestamp']) != str:
    candle['timestamp'] = iso_str(candle['timestamp'])
  return candle

def add(bin, candle):
  '''REST function: add candle.'''
  uri=f'{DATA_URI}/candles'
  candle = transform(bin, candle)

  # Note: _id validation here is not very sophisticated
  if '_id' in candle and type(candle['_id']) == str:
    s.put(uri, json=candle)
  else:
    s.post(uri, json=candle)

def to_dict(df):
  '''Convert dataframe to dict before sending them off to the database.'''
  return df.reset_index().replace({ nan: None }).to_dict(orient='records')

def add_dataframe(df, bin, func=add, suppress=True):
  '''Semi-REST function: add dataframe.'''

  # Disable logging here to suppress the requests module's spam
  if suppress: logging.disable(logging.CRITICAL)
  for candle in to_dict(df): func(bin, candle)
  if suppress: logging.disable(logging.NOTSET)

def add_forecast(bin, candle, level):
  '''REST function: add forecast.'''
  candle = transform(bin, candle)
  candle = {k : v for k, v in candle.items() if v is not None}
  candle['level'] = level
  s.put(f'{DATA_URI}/forecast?timeframe={bin}&level={level}', json=candle)

def add_dataframe_forecast(df, bin, level, suppress=True):
  '''Semi-REST function: add forecast dataframe.'''
  f = partial(add_forecast, level=level)
  add_dataframe(df, bin, func=f, suppress=suppress)

def clear_forecasts():
  s.delete(f'{DATA_URI}/forecast')

def iso_str(dt):
  '''Auxiliary function: unparse datetime object to ISO string.'''
  return dt.isoformat().replace('+00:00', '.000Z')

def parseTimestamps(data):
  '''Auxiliary function: parse timestamps inside a dict.'''
  for row in data:
    row['timestamp'] = isoparse(row['timestamp'])
  return data

# Indicators
INDICATORS = [
  partial(sma, period=10),
  partial(sma, period=200),
  
  partial(ema, period=21),
  partial(ema, period=55),
  partial(ema, period=89),
  partial(ema, period=200),
  partial(ema, period=377),
  
  stoch,
  stoch_K,
  stoch_D,
  
  change_up,
  change_down,
  rma_up,
  rma_down,
  rsi,

  hv,
  hvp,
  hvp_ma,
]
PART_INDS = [
  partial(sma_part, period=10),
  partial(sma_part, period=200),
  
  partial(ema_part, period=21),
  partial(ema_part, period=55),
  partial(ema_part, period=89),
  partial(ema_part, period=200),
  partial(ema_part, period=377),
  
  stoch_part,
  stoch_K_part,
  stoch_D_part,
  
  change_up_part,
  change_down_part,
  rma_up_part,
  rma_down_part,
  rsi_part,

  hv_part,
  hvp_part,
  hvp_ma_part,
]

# Forecasting
bs = BlackScholes()
DEPTH = 6
ANNUAL = 100
# corresponding probabilities for reverse computations
SD_LVLS = [.5 - lvl / 2 for lvl in [.68, .95, .997]]
TOUCH_LVLS = [lvl / 2 for lvl in SD_LVLS]
ANNUALIZE = {
  '1h':  24 * ANNUAL,
  '2h':  12 * ANNUAL,
  '3h':  8  * ANNUAL,
  '4h':  6  * ANNUAL,
  '6h':  4  * ANNUAL,
  '12h': 2  * ANNUAL,
  '1d':       ANNUAL,
  '2d':       ANNUAL / 2,
  '3d':       ANNUAL / 3,
  '1w':       ANNUAL / 7,
  '1M':       ANNUAL / 30.4167,
  '3M':       ANNUAL / 91.25,
}
# amount of rows to consider when applying indicators to estimated future
# prices, this value should be greater than any period of any indicator applied
FORECAST_CUTOFF = 377

class OhlcTracker(Observer):
  '''Tracks historical and partial data.'''

  def __init__(self, resample=True, insert=True):
    self.resample = resample
    self.insert = insert
    self.data = {}
    self.parts = {}
    self.forecasts = {}
    self.fetched_source = False
    self.fetched_resamples = False

  def build(self):
    '''Build original and resampled dataframes. Update on repeated calls. Apply indicators. Insert into database.'''
    with LOCK:
      action = 'UPDATE' if self.fetched_source else 'BUILD'
      logger.info(f'========== {action} STARTED')
      logger.info('========== PHASE 1 ===== FETCH & PARSE')
      logger.info('========== PHASE 1.1 === SOURCE DATA')
      self.fetch_source()
      self.fetched_source = True
      if self.resample:
        logger.info('========== PHASE 1.2 === RESAMPLE DATA')
        self.fetch_resamples()
        self.fetched_resamples = True
      logger.info('========== PHASE 2 ===== APPLY & UPSERT')
      self.apsert()
      logger.debug('========== PHASE 3 ===== PREPARE FORECASTS')
      self.prep_forecasts()
      logger.info(f'========== {action} ENDED')
      return self

  def fetch_source(self):
    '''Fetch missing historical data.'''
    for bin in BINS:
      
      # Fetch previous data into dataframe
      if not self.fetched_source:
        self.fetch_data(bin)

      # Fetch starting point  
      start = 1 + count(bin)
      logger.info(f'Updating {bin} data (start from {start}).')

      # Fetch historical data
      while True:
        t0 = time()
        
        # Remote fetch from BitMEX
        data = parseTimestamps(client.rest.get_buckets(bin, start, count=COUNT))
        if len(data) < COUNT:
          logger.debug(f'[{time() - t0}] BitMEX response: {data}')

        # Concatenate new data to dataframe
        df = pd.DataFrame(data)
        if len(data):
          df = df.set_index('timestamp')
        df = pd.concat([self.data[bin], df], sort=False)
        self.data[bin] = df
        logger.info(f'{bin} candles: {len(df)} (+{len(data)})')

        # Stop condition (nothing left to fetch)
        if not data: break

        # Miscellaneous bureaucracy
        start += COUNT
        dur = TIMEOUT - time() + t0
        if dur > 0: 
          logger.debug(f'Sleeping for {dur}')
          sleep(dur)

      # Fetch partial
      self.parts[bin] = client.rest.get_buckets(bin, start, partial=True)[0]

  def fetch_resamples(self):
    '''Resample data from original data.'''
    for bin in BINS:
      for dbin in DBINS[bin]:
        
        # Fetch previous data into dataframe
        if not self.fetched_resamples:
          self.fetch_data(dbin)
        df = self.data[dbin]
        
        # Determine resampling starting point
        row = 0
        if not df.empty:
          lt = df.index[-1]
          ft, n = first_timestamp(bin, dbin, t=self.data[bin].index[0])
          row = int(n + (lt - ft) / TIMEDELTAS[bin] + OFFSETS[bin])
        
        # Resample bin into dbin
        if row < len(self.data[bin]):
          logger.info(f'Resampling {bin} data into {dbin} data...')
          new_rows = self.data[bin].iloc[row:]
          data, self.parts[dbin] = resample(new_rows, bin, dbin)
          df = pd.concat([df, data], sort=False)
          self.data[dbin] = df
          logger.info(f'{dbin} candles: {len(df)} (+{len(data)})')

          # Update resampled partial with original's partial
          self.parts[dbin]['high'] = max(self.parts[bin]['high'],
                                         self.parts[dbin]['high'])
          self.parts[dbin]['low'] = min(self.parts[bin]['low'],
                                        self.parts[dbin]['low'])
          self.parts[dbin]['close'] = self.parts[bin]['close']
          
          # Unparse a partial's timestamp to save time when upserting to db
          self.parts[dbin]['timestamp'] = iso_str(self.parts[dbin]['timestamp'])

        # Case where no resampling is necessary (so no candles to insert either)
        else:
          logger.info(f'No need to resample {bin} data into {dbin} data')
          logger.info(f'{dbin} candles: {len(self.data[dbin])} (+0)')
          self.parts[dbin] = dict(self.parts[bin]) # create copy

          # [Bug fix] adjust the timestamp
          t = isoparse(self.parts[bin]['timestamp'])
          ft = first_timestamp(bin, dbin, t)[0]
          self.parts[dbin]['timestamp'] = iso_str(ft)
          logger.debug(f'Set {dbin} timestamp to {ft}')

  def fetch_data(self, bin):
    '''Transform dict into pandas DataFrame.'''
    logger.info(f'Fetching {bin} data...')
    self.data[bin] = pd.DataFrame(get(bin)).replace({ None: nan })
    if not self.data[bin].empty:
      self.data[bin].set_index('timestamp', inplace=True)

  def apsert(self):
    '''Apply indicators and insert into database.'''
    for bin, df in self.data.items():
      t0 = time()

      # Apply indicators
      logger.info(f'Applying indicators to {bin}...')
      first_aff_row = self.apply_indicators(self.data[bin])
      logger.debug(f'[{time() - t0}] First affected {first_aff_row}/{len(df)}')

      # Upsert into database
      if self.insert:
        aff_df = df.loc[df.index[first_aff_row:]]
        logger.info(f'Upserting {len(aff_df)} affected {bin} candles...')
        if not aff_df.empty:
          add_dataframe(aff_df, bin)
          logger.debug(f'[{time() - t0}] Upsertion done.')
      else:
        logger.info('Not upserting into database.')

  def apply_indicators(self, df):
    '''Add all indicator columns to a dataframe.'''
    return min(indicator(df) for indicator in INDICATORS)

  def prep_forecasts(self):
    '''Prepare forecasts by precomputing the volatility and timestamps to feed into the reverse computations. Also clear all forecasts on the data server. That is the easiest way to get rid of the outdated forecasts which wouldn't be overwritten anymore.'''
    self.times = {}
    for bin in self.data:
      self.times[bin] = []
      t = self.data[bin].index[-1]
      for depth in range(1, 1 + DEPTH):
        self.times[bin].append(next_timestamp(bin, t, depth))
    self.clear_forecasts = True

  def update(self, observable, message):
    '''This method is to be called by the socket's message notifier.'''

    # Update historical data
    if message['table'] == 'tradeBin1h' or message['table'] == 'tradeBin1d':
      if message['action'] == 'insert':
        self.build()

    # Update partial HLC data
    elif message['table'] == 'trade':
      for datum in message['data']:
        price = datum['price']
        for bin in self.parts:
          self.parts[bin]['close'] = price
          if price > self.parts[bin]['high']:
            self.parts[bin]['high'] = price
          elif price < self.parts[bin]['low']:
            self.parts[bin]['low'] = price

  def run(self):
    '''Start updating indicators, upserting partials and upserting forecasts in a separate thread.'''

    def partial_loop():
      def upsert_partial(bin, suppress=True):
        if suppress:
          logging.disable(logging.CRITICAL)

        partial = dict(self.parts[bin])
        for k, v in partial.items():
          if isinstance(v, Number) and isnan(v):
            partial[k] = None
        s.put(f'{DATA_URI}/partials?timeframe={bin}', json=partial)

        if suppress:
          logging.disable(logging.NOTSET)

      while self.running:
        t0 = time()
        with LOCK:
          try:
            for bin in self.parts:

              # Update partial indicators
              for part_ind in PART_INDS:
                part_ind(self.data[bin], self.parts[bin])

              # Upsert into db
              if self.insert:
                upsert_partial(bin)
          except Exception as e:
            logger.warning(f'Warning: exception in partial loop. {e}')

        # Limit iterations
        sleep(max(PARTIAL_DAEMON_TIMEOUT - time() + t0, 0))

    def forecast_loop():

      # Auxiliary lambda
      ohlc = lambda t, o, h, l, c: {
        'timestamp': t, 'open': o, 'high': h, 'low': l, 'close': c
      }

      def compute_forecast(bin):
        self.forecasts[bin] = []

        # Compute time remaining to each next tick
        annualize = lambda dt: dt.total_seconds() / (86400 * ANNUAL)
        now = datetime.utcnow().replace(tzinfo=UTC)
        rems = [annualize(time - now) for time in self.times[bin]]

        # Compute volatility
        volatility = self.data[bin]['hv'][-1] * sqrt(ANNUALIZE[bin])

        # Forecast the expected candles ("0th standard deviation level")
        c = self.parts[bin]['close']
        expected = []
        for rem, time in zip(rems, self.times[bin]):
          low, high = bs.reverse(.25, c, volatility, rem)
          expected.append(ohlc(time, c, high, low, c))
        levels = deque([expected])
        
        # Forecast candles at 1st, 2nd and 3rd standard deviation levels
        for prob, touch_prob in zip(SD_LVLS, TOUCH_LVLS):
          lower, upper = [], []
          prev_l, prev_u = c, c
          for rem, time in zip(rems, self.times[bin]):
            l_close, u_close = bs.reverse(prob, c, volatility, rem)
            low, high = bs.reverse(touch_prob, c, volatility, rem)
            lower.append(ohlc(time, prev_l, prev_l, low, l_close))
            upper.append(ohlc(time, prev_u, high, prev_u, u_close))
            prev_l, prev_u = l_close, u_close
          levels.appendleft(lower)
          levels.append(upper)

        # Apply indicators to forecast prices
        for level in levels:
          proxy = pd.DataFrame(level).set_index('timestamp')

          # This optimization significantly cuts computation time
          # (-50% at the time)
          real = self.data[bin]
          real = real.loc[real.index[-FORECAST_CUTOFF:]]

          proxy = real.append(proxy, sort=False)
          self.apply_indicators(proxy)
          self.forecasts[bin].append(proxy.tail(DEPTH))

        # Upsert forecast to db
        for level, df in enumerate(self.forecasts[bin]):
          add_dataframe_forecast(df, bin, level)

      while self.running:
        t0 = time()
        with LOCK:
          if self.clear_forecasts:
            clear_forecasts()
            self.clear_forecasts = False
          try:
            for bin in self.data:
              compute_forecast(bin)
          except Exception as e:
            logger.warning(f'Warning: exception in forecast loop. {e}')

        # Limit iterations
        sleep(max(FORECAST_DAEMON_TIMEOUT - time() + t0, 0))

    self.running = True
    start_daemon(target=partial_loop)
    start_daemon(target=forecast_loop)

if __name__ == '__main__':
  DEFAULT_VERBOSITY = logging.INFO

  # Add options
  parser = ArgumentParser()
  parser.add_argument('-c', '--do-not-connect', action='store_true',
                      help='Don\'t connect to BitMEX\' websocket.',
                      default=False)
  parser.add_argument('-r', '--do-not-resample', action='store_true',
                      help='Don\'t resample bins.',
                      default=False)
  parser.add_argument('-i', '--do-not-insert', action='store_true',
                      help='Don\'t insert into database.',
                      default=False)
  parser.add_argument('-v', '--verbosity', type=int,
                      help='Verbosity level, INFO=20, DEBUG=10, ...',
                      default=DEFAULT_VERBOSITY)
  args = parser.parse_args()

  logger.setLevel(args.verbosity)
  if args.verbosity != DEFAULT_VERBOSITY:
    logger.info(f'[-v flag]: Verbosity level set to {args.verbosity}.')
  if args.do_not_connect:
    logger.info('[-c flag]: Not connecting to BitMEX\' websocket.')
  if args.do_not_resample:
    logger.info('[-r flag]: Not resampling bins.')
  if args.do_not_insert:
    logger.info('[-i flag]: Not inserting data.')

  ohlct = OhlcTracker(resample=not args.do_not_resample,
                      insert=not args.do_not_insert)
  ohlct.build()

  # Setup websocket and connect to it
  client.websocket.symbols = ['XBTUSD']
  client.websocket.subscriptions = {
    'execution': False,
    'instrument': False,
    'order': False,
    'orderBookL2': False,
    'position': False,
    'quote': False,
    'trade': True,
    'margin': False,
    'tradeBin1m': False,
    'tradeBin5m': False,
    'tradeBin1h': True,
    'tradeBin1d': True,
  }
  client.websocket.message_notifier.add_observer(ohlct)
  client.websocket.ready_notifier.add_observer(Starter(ohlct))
  client.websocket.close_notifier.add_observer(Stopper(ohlct))

  if not args.do_not_connect:
    client.connect()
