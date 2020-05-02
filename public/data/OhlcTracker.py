from BitmexClient import BitmexClient
from Observer import Observer
from daemon import start_daemon, Starter, Stopper
from indicators import ema, stoch, rsi_ema, smooth_stoch, hvp, part_ind
from quick_write import quick_write, quick_read

from sys import stdout
from os.path import exists
from functools import partial
from time import time, sleep, mktime
from json import loads, dumps
from datetime import datetime, timedelta
from dateutil.parser import isoparse
from dateutil.relativedelta import relativedelta
from argparse import ArgumentParser
import requests
import pandas as pd

#--- MIRRORED

from pprint import PrettyPrinter
pp = PrettyPrinter()

# Logger setup
import logging
logger = logging.getLogger()
if __name__ == '__main__':
  ch = logging.StreamHandler(stdout)
  formatter = logging.Formatter("%(asctime)s: %(message)s")
  ch.setFormatter(formatter)
  logger.addHandler(ch)

client = BitmexClient(key=None, secret=None)

#--- END MIRRORED

# RESTful candle data API
DATA_SERVER = 'http://localhost:3001/candles'
s = requests.Session()

# Filename constants
filename_partial = lambda bin: f'ohlc-{bin}-partial.json'

# Runtime constants
BINS = ['1h', '1d']
COUNT = 1000# results to fetch per request (max is 1000)
TIMEOUT = 2# seconds, to be safe
DAEMON_TIMEOUT = 1# second

# Derived bins (DBINS)
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
    return result, close_candle(first_timestamp(bin, dbin, df.index[0])[0])

  # 
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

# Indicators
INDICATORS = {
  'ema-21': partial(ema, period=21),
  'ema-55': partial(ema, period=55),
  'ema-89': partial(ema, period=89),
  'ema-200': partial(ema, period=200),
  'stoch': smooth_stoch,
  'rsi': rsi_ema,
  'hvp': hvp,
}

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

# TODO wrap my head around these. These were "derived experimentally"
OFFSETS = {
  '1h': -1,
  '1d': 1,
}

def first_timestamp(bin, dbin, t):
  '''Find the first valid timestamp of a derived bin.'''
  n = 0
  while not DCONDS[bin][dbin](t):
    n += 1
    t += TIMEDELTAS[bin]
  return t, n

# REST function: get candles
get = lambda bin: parseTimestamps(s.get(f'{DATA_SERVER}?timeframe={bin}').json())

def add(candle, bin):
  '''REST function: add candle.'''
  candle = dict(candle) # makes (local) copy
  candle['timeframe'] = bin
  if type(candle['timestamp']) != str:
    candle['timestamp'] = iso_str(candle['timestamp'])
  s.post(f'{DATA_SERVER}/add', json=candle)

def add_dataframe(df, bin):
  '''semi-REST function: add dataframe.'''
  df = df.reset_index()
  df['timestamp'] = df['timestamp'].apply(iso_str)
  for candle in df.to_dict(orient='records'):
    add(candle, bin)

# Auxiliary function: datetime object to ISO string
iso_str = lambda dt: dt.isoformat().replace('+00:00', '.000Z')

def parseTimestamps(data):
  '''Auxiliary function: parse timestamps.'''
  for row in data:
    row['timestamp'] = isoparse(row['timestamp'])
  return data

class OhlcTracker(Observer):
  '''Tracks historical and partial data.'''
  def __init__(self, derive=True):
    self.logger = logging.getLogger(__name__)

    self.derive = derive
    self.dataframes = {}
    self.parts = {}
    
    # self.build()

  #
  def build(self, repeat=False):
    self.build_orig(repeat)
    if self.derive:
      self.build_deriv(repeat)

  def build_orig(self, repeat=False):
    '''Fetch historical data, insert diff into database, write partials to file.'''
    for bin in BINS:
      
      # Fetch previous data, init dataframe, starting point implied
      if not repeat:
        self.build_dataframe(bin)
      start = len(self.dataframes[bin])
      self.logger.info(f'Updating {bin} data (start from {start}).')

      # Fetch historical data
      while True:
        t0 = time()
        
        # Remote work: fetch from BitMEX
        data = parseTimestamps(client.rest.get_buckets(bin, start, count=COUNT))
        if len(data) < COUNT:
          self.logger.debug(f'[{time() - t0}] BitMEX response: {data}')

        # Stop condition (nothing left to fetch)
        if not data: break 

        # Local work: append to dataframes and data
        df = pd.DataFrame(data).set_index('timestamp')
        self.dataframes[bin] = pd.concat([self.dataframes[bin], df], sort=False)
        self.logger.debug(f'{bin}: {len(self.dataframes[bin])} (+{len(data)})')

        # Local work: database insertion
        self.logger.debug(f'[{time() - t0}] Inserting into database...')
        for candle in data:
          add(candle, bin)

        # Miscellaneous bureaucracy
        start += COUNT
        dur = TIMEOUT - time() + t0
        if dur > 0: 
          self.logger.debug(f'Sleeping for {dur}')
          sleep(dur)

      # Fetch partial
      self.parts[bin] = client.rest.get_buckets(bin, start, partial=True)[0]
      self.parts[bin]['timestamp'] = isoparse(self.parts[bin]['timestamp'])

  def build_deriv(self, repeat=False):
    '''Derive data.'''
    for bin in BINS:
      for dbin in DBINS[bin]:
        if logger.level >= logging.DEBUG: print()
        
        # Get previous data from database, init dataframe
        if not repeat:
          self.build_dataframe(dbin)
        
        # Determine resample starting point
        row = 0
        if not self.dataframes[dbin].empty:
          lt = self.dataframes[dbin].index[-1]
          ft, n = first_timestamp(bin, dbin, t=self.dataframes[bin].index[0])
          row = n + (lt - ft) / TIMEDELTAS[bin] + OFFSETS[bin]
          assert int(row) == row
          row = int(row)
          
          self.logger.debug(f'  ft  {ft}')
          self.logger.debug(f'  n   {n}')
          self.logger.debug(f'  lt  {lt}')
          self.logger.debug(f'  row {row}')
          if row == len(self.dataframes[bin]): self.logger.debug(f'      <no timestamp at row {row}>')
          elif row > len(self.dataframes[bin]): self.logger.debug(f'      <CRITICAL ERROR: tried to read timestamp at row {row}>')
          else: self.logger.debug(f"      {self.dataframes[bin].index[row]}")
        self.logger.debug(f'Deriving {dbin} (from {row}/{len(self.dataframes[bin])} total {len(self.dataframes[bin])-row} rows)')

        # Resample bin into dbin
        if row < len(self.dataframes[bin]):
          data, self.parts[dbin] = resample(self.dataframes[bin].iloc[row:], bin, dbin)
          self.dataframes[dbin] = pd.concat([self.dataframes[dbin], data], sort=False)

          # Update derived partial with original's partial
          self.parts[dbin]['high'] = max(self.parts[bin]['high'],
                                         self.parts[dbin]['high'])
          self.parts[dbin]['low'] = min(self.parts[bin]['low'],
                                        self.parts[dbin]['low'])
          self.parts[dbin]['close'] = self.parts[bin]['close']

          # Insert new candles into database
          self.logger.debug(f'adding \n{str(data)}')
          if not data.empty:
            add_dataframe(data, dbin)

        # Case where no resampling is necessary (so no candles to insert either)
        else:
          self.parts[dbin] = self.parts[bin]

        self.logger.debug(f'{dbin} partial {str(self.parts[dbin])}')

  def build_dataframe(self, bin):
    '''TODO comment.'''
    self.dataframes[bin] = pd.DataFrame(get(bin))
    if not self.dataframes[bin].empty:
      self.dataframes[bin].set_index('timestamp', inplace=True)

  def update(self, observable, message):
    '''This method is to be called by the socket's message notifier.'''

    # Update historical data
    if message['table'] == 'tradeBin1h' or message['table'] == 'tradeBin1d':
      if message['action'] == 'insert':
        self.build(repeat=True)

    # Update partial data
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
    '''Start updating indicators and writing partials.'''

    def quick_write_partial(bin):
      partial = dict(self.parts[bin])
      partial['timestamp'] = iso_str(partial['timestamp'])
      quick_write(filename_partial(bin), dumps(partial))

    def loop():
      while self.running:
        for bin in self.parts:
          quick_write_partial(bin)
        sleep(DAEMON_TIMEOUT)

    self.running = True
    start_daemon(target=loop)

DEFAULT_VERBOSITY = logging.DEBUG
if __name__ == '__main__':

  # Add options
  parser = ArgumentParser()
  parser.add_argument('-c', '--do-not-connect', action='store_true',
                      help='Don\'t connect to BitMEX\' websocket.')
  parser.add_argument('-d', '--do-not-derive', action='store_true',
                      help='Don\'t derive bins.', default=False)
  parser.add_argument('-v', '--verbosity', type=int,
                      help='Verbosity level, DEBUG=10, INFO=20.',
                      default=DEFAULT_VERBOSITY)
  args = parser.parse_args()

  logger.setLevel(args.verbosity)
  if args.verbosity != DEFAULT_VERBOSITY:
    logger.info(f'[-v flag]: Verbosity level set to {args.verbosity}.')
  if args.do_not_connect:
    logger.info('[-c flag]: Not connecting to BitMEX\' websocket.')
  if args.do_not_derive:
    logger.info('[-d flag]: Not deriving bins.')

  ohlct = OhlcTracker(derive=not args.do_not_derive)
  ohlct.build()

  # Setup websocket and connect to it
  if not args.do_not_connect:
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
    client.connect()

  # TODO remove debug tools
  df=ohlct.dataframes['1h']