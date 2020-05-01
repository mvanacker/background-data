from BitmexClient import BitmexClient
from Observer import Observer
from quick_write import quick_write, quick_read
from daemon import start_daemon, Starter, Stopper
from indicators import ema, stoch, rsi_ema, smooth_stoch, hvp, part_ind

from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import HTTPError
from sys import stdout
from os.path import exists
from datetime import datetime, timedelta
from time import sleep, mktime
from json import loads, dumps
from dateutil.parser import isoparse
from dateutil.relativedelta import relativedelta
from functools import partial
import pandas as pd
from argparse import ArgumentParser
# import numpy as np

#--- MIRRORED

from pprint import PrettyPrinter
pp = PrettyPrinter()

import logging
# Logger setup
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)# Change this to DEBUG if you want a lot more info
ch = logging.StreamHandler(stdout)
formatter = logging.Formatter("%(asctime)s: %(message)s")
ch.setFormatter(formatter)
if __name__ == '__main__':
  logger.addHandler(ch)

client = BitmexClient(key=None, secret=None)

#--- END MIRRORED

# Filename constants
FILENAME = lambda bin: f'ohlc-{bin}.json'
FILENAME_PARTIAL = lambda bin: f'ohlc-{bin}-partial.json'
FILENAME_INDICATOR = lambda bin, ind: f'ohlc-{bin}-{ind}.json'
FILENAME_PART_IND = lambda bin, ind: f'ohlc-{bin}-{ind}-partial.json'

# Runtime constants
BINS = ['1h', '1d']
COUNT = 1000# results to fetch per request (max is 1000)
# COLUMNS = 'symbol,timestamp,open,high,low,close'
# KEYS = COLUMNS.split(',')
# MAX_DF_LEN = 1000# rows
TIMEOUT = 2# seconds, to be safe

# Derived bins (DBINS)
to_dt = lambda candle: isoparse(candle['timestamp'])
to_tt = lambda candle: to_dt(candle).timetuple()
condition_hourly = lambda candle, h: to_dt(candle).hour % h == 0
condition_daily_mr = lambda candle, m, r: mktime(to_tt(candle)) % m == r
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
    '1w': lambda candle: to_tt(candle).tm_wday == 0,
    '1M': lambda candle: to_tt(candle).tm_mday == 1,
    '3M': lambda c: to_tt(c).tm_mday == 1 and to_tt(c).tm_mon % 3 == 1,
  },
}

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

# Indicator updating constants
DAEMON_TIMEOUT = 300# seconds

# Auxiliary REST call function
DATA_SERVER = 'http://localhost:3001/candles/'
def query(self, method, function, params={}):
  params = urlencode(params)
  url = f'{DATA_SERVER}{function}?{params}'
  try:
    request = Request(url=url, method=method)
    with urlopen(request) as f:
      return loads(f.read())
  except HTTPError as e:
    return loads(e.read())

# Auxiliary function: filter out garbage fields
filter = lambda msg: msg['data'][0]#{k:d for k, d in msg['data'][0].items()}#= if k in KEYS}

# Auxiliary function: datetime object to ISO string
to_iso_str = lambda dt: dt.isoformat().replace('+00:00', '.000Z')

# Auxiliary function: resample time series TODO learn pandas
def resample(series, condition):
  result = []
  is_opening = True
  open = series[0]['open']
  high = series[0]['high']
  low = series[0]['low']
  close = series[0]['close']
  for candle in series:

    # Update candle
    if is_opening:
      if not 'open' in candle:
        logger.debug(dumps(candle))
      open = candle['open']
      high = candle['high']
      low = candle['low']
      is_opening = False
    high = max(high, candle['high'])
    low = min(low, candle['low'])
    close = candle['close']

    # Candle completed
    if condition(candle):
      result.append({
        'symbol': candle['symbol'],
        'timestamp': candle['timestamp'],
        'open': open,
        'high': high,
        'low': low,
        'close': close,
      })
      is_opening = True

  # Assuming 2 or more candles that are equidistant on the time axis
  t0 = isoparse(result[-1]['timestamp'])
  t1 = isoparse(result[-2]['timestamp'])
  partial_candle = {
    'timestamp': to_iso_str(t0 + (t0 - t1)),
    'symbol': result[-1]['symbol'],
    'open': close if is_opening else open,
    'high': close if is_opening else high,
    'low': close if is_opening else low,
    'close': close if is_opening else close,
  }

  return result, partial_candle

# Auxiliary auxiliary function: reset DataFrame
def reset_df(df):
  rdf = df.reset_index()
  rdf['timestamp'] = rdf['timestamp'].apply(to_iso_str)
  return rdf
# Auxiliary function: convert DataFrame to JSON
df_to_json = lambda df: reset_df(df).to_json(orient='records')
# Auxiliary function: convert DataFrame to dict
df_to_dict = lambda df: reset_df(df).to_dict(orient='records')

# Auxiliary function: compute next timestamp
# Note: this function is called after the condition is already met, so there is
# no need to consider leap seconds.
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
next_timestamp = lambda bin, t: to_iso_str(isoparse(t) + TIMEDELTAS[bin])

# Tracks historical and partial data
class OhlcTracker(Observer):
  def __init__(self, derive=True):
    self.logger = logging.getLogger(__name__)

    # Original data
    self.data = {}
    self.partials = {}
    for bin in BINS:

      # Determine fetch starting point
      filename = FILENAME(bin)
      if not exists(filename):
        self.data[bin] = []
        start = 0
        verb = 'Creating'
      else:
        self.data[bin] = loads(quick_read(filename))
        start = len(self.data[bin])
        verb = 'Updating'
      self.logger.info(f'{verb} {bin} file (start from {start}).')

      # Create or update historical data
      while True:
        res = client.rest.get_buckets(bin, start, count=COUNT)
        self.logger.debug(f'BitMEX response: {res}')
        if not res: break
        self.data[bin] += res
        self.logger.debug(f'{bin}: {len(self.data[bin])} (+{len(res)})')
        start += COUNT
        sleep(TIMEOUT)
      self.quick_write_data(bin)

      # Create or update partials
      self.partials[bin] = client.rest.get_buckets(bin, start, partial=True)[0]
      self.quick_write_partial(bin)

    if derive:
      # Derived data
      for bin in BINS:
        for dbin in DBINS[bin]:
          self.logger.info(f'Deriving {dbin} bin...')
          self.data[dbin], self.partials[dbin] = resample(self.data[bin],
                                                          DCONDS[bin][dbin])
          self.quick_write_data(dbin)
          self.quick_write_partial(dbin)

      # Update derived partials
      for bin in BINS:
        for dbin in DBINS[bin]:
          self.partials[dbin]['high'] = max(self.partials[dbin]['high'],
                                            self.partials[bin]['high'])
          self.partials[dbin]['low'] = min(self.partials[dbin]['low'],
                                          self.partials[bin]['low'])
          self.partials[dbin]['close'] = self.partials[bin]['close']

    # Convert to Pandas DataFrames
    self.logger.info('Building historical dataframes...')
    self.dataframes = {}
    for bin in self.data:
      self.build_dataframe(bin)

    # Build indicator history
    self.logger.info('Building indicator history...')
    self.indicators = {}
    for bin in self.data:
      self.indicators[bin] = {}
    self.apply_indicators()

    # Initiate partial indicators
    self.part_inds = {}
    for bin in self.data:
      self.part_inds[bin] = {}
    self.update_part_inds()

  # Convert initial JSON data to Pandas DataFrame
  def build_dataframe(self, bin):
    df = pd.DataFrame(self.data[bin])
    df['timestamp'] = df['timestamp'].apply(isoparse)
    self.dataframes[bin] = df.set_index('timestamp')
    # self.dataframes[bin] = df[-MAX_DF_LEN:]

  # Indicator updating daemon; it may actually be computationally feasible to
  # update indicators on every tick, I sort of initially assumed it wouldn't.
  # It does sound like an attractive idea to update indicators in a separate
  # thread. TODO: just maybe more frequent than once per second?
  def run(self):
    self.running = True
    start_daemon(target=self.update_inds_write_parts)

  # Convenience methods
  def quick_write_data(self, bin):
    quick_write(FILENAME(bin), dumps(self.data[bin]))
  def quick_write_partial(self, bin):
    quick_write(FILENAME_PARTIAL(bin), dumps(self.partials[bin]))
  def quick_write_indicator(self, bin, ind):
    quick_write(FILENAME_INDICATOR(bin, ind),
                df_to_json(self.indicators[bin][ind]))
  def quick_write_part_ind(self, bin, ind):
    quick_write(FILENAME_PART_IND(bin, ind),
                df_to_json(self.part_inds[bin][ind]))

  # This method is to be called by the socket's message notifier
  def update(self, observable, message):

    # Update historical data
    if message['table'] == 'tradeBin1h' and message['action'] == 'insert':
      self.update_history('1h', message)
    elif message['table'] == 'tradeBin1d' and message['action'] == 'insert':
      self.update_history('1d', message)

    # Update partials
    elif message['table'] == 'trade':
      for datum in message['data']:
        self.update_partial(datum['price'])

  # Update historical data: a bin and all bins that derive from it
  def update_history(self, bin, message):
    # candle = filter(message)
    candle = message['data'][0]
    self.update_single(bin, candle)
    for dbin in [dbin for dbin in self.data if dbin in DBINS[bin]]:
      if DCONDS[bin][dbin](candle):
        self.partials[dbin]['close'] = candle['close'] # correction
        self.update_single(dbin, self.partials[dbin])

  # Update historical data: single bin (original or derived)
  def update_single(self, bin, candle):
    self.data[bin].append(candle)
    self.quick_write_data(bin)
    self.logger.debug(f'Updated {bin}: {dumps(candle)}')
    self.build_dataframe(bin)
    self.logger.debug(f'Updated {bin} dataframe')
    self.apply_indicators_single(bin)
    self.logger.debug(f'Applied indicators to {bin} dataframe')

    # Reset partial
    self.partials[bin]['timestamp'] = next_timestamp(bin, candle['timestamp'])
    self.logger.debug(f"Next {bin} timestamp {self.partials[bin]['timestamp']}")
    self.partials[bin]['open'] = candle['close']
    self.partials[bin]['high'] = candle['close']
    self.partials[bin]['low'] = candle['close']
    self.partials[bin]['close'] = candle['close']

  # Update partial data
  def update_partial(self, price):
    for bin in self.partials:
      self.partials[bin]['close'] = price
      if price > self.partials[bin]['high']:
        self.partials[bin]['high'] = price
      elif price < self.partials[bin]['low']:
        self.partials[bin]['low'] = price

  # Apply each indicator to one bin
  def apply_indicators_single(self, bin):
    for name, ind in INDICATORS.items():
      self.indicators[bin][name] = ind(self.dataframes[bin]) 
      self.quick_write_indicator(bin, name)

  # Apply each indicator to every bin
  def apply_indicators(self):
    for bin in self.data:
      self.apply_indicators_single(bin)

  # Update partial indicators
  def update_part_inds(self):
    for bin in self.data:
      for name, ind in INDICATORS.items():
        self.part_inds[bin][name] = part_ind(self.dataframes[bin],
                                             self.partials[bin], ind)

  # To be run as daemon
  def update_inds_write_parts(self):
    while self.running:
      self.update_part_inds()
      for bin in self.data:
        self.quick_write_partial(bin)
        for name in INDICATORS:
          self.quick_write_part_ind(bin, name)
      sleep(DAEMON_TIMEOUT)

if __name__ == '__main__':

  # Add option to not connect to socket
  parser = ArgumentParser()
  parser.add_argument('-c', '--do-not-connect', action='store_true',
                      help='Don\'t connect to BitMEX\' websocket.')
  parser.add_argument('-d', '--do-not-derive', action='store_true',
                      help='Don\'t derive bins.', default=False)
  args = parser.parse_args()

  if args.do_not_connect:
    logger.info('[-c flag]: Not connecting to BitMEX\' websocket.')
  if args.do_not_derive:
    logger.info('[-d flag]: Not deriving bins.')

  # Initialize OHLC tracker (computes and writes historical data)
  ohlct = OhlcTracker(derive=not args.do_not_derive)

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