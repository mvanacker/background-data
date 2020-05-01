from BitmexClient import BitmexClient
from Observer import Observer
# from OhlcTracker import OhlcTracker
from polling import poll_fng, poll_cmc, poll_deribit
from quick_write import quick_write
from daemon import start_daemon, Stopper

from sys import stdout
from statistics import mean
from collections import deque
from dateutil.parser import parse
from datetime import timedelta, datetime
from threading import Thread, Lock
from time import sleep, time
from json import loads, dumps
from math import nan, isnan
from threading import Thread

### Debug

from pprint import PrettyPrinter
pp = PrettyPrinter()

import logging
# Logger setup
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)# Change this to DEBUG if you want a lot more info
ch = logging.StreamHandler(stdout)
formatter = logging.Formatter("%(asctime)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

### End Debug

# Constants
LOCK = Lock()
VF_MA_LEN = 10 # Volume Flow Moving Average Length (in seconds)
VF_MA_LEN_TD = timedelta(seconds=VF_MA_LEN)
VF_HIST_LEN = 300 # Volume Flow History Length (in seconds)
RECENT_TRADES_LEN = 100
RECENT_TRADES_MIN_SIZE = 100000
TIMEOUT = 1# (in seconds)

# File Constants
PRICE_FILE = 'price.txt'
BUY_FLOW_FILE = 'buy-flow.txt'
SELL_FLOW_FILE = 'sell-flow.txt'
PRICE_HISTORY_FILE = 'price-history.json'
BUY_FLOW_HISTORY_FILE = 'buy-flow-history.json'
SELL_FLOW_HISTORY_FILE = 'sell-flow-history.json'
RECENT_TRADES_FILE = 'recent-trades.json'
OPEN_INTEREST_FILE = 'open-interest.txt'
OPEN_INTERESTS_FILE = 'open-interest.json'
OPEN_INTEREST_HISTORY_FILE = 'open-interest-history.json'

# Measures volume flow
class VolumeFlowMeasurer(Observer):
  def __init__(self, futures=[]):

    # Volume flow fields
    self.price = nan
    self.price_history = deque()
    self.trades = deque() # trades within measurement window
    self.buy_flow_history = deque()
    self.sell_flow_history = deque()
    self.recent_trades = deque() # a constant amount of recent trades

    # Open interest fields
    self.futures = futures
    self.open_interests = {i:0 for i in futures}
    self.open_interest = nan
    self.open_interest_history = deque()

    # Start measurement thread
    self.running = True
    start_daemon(target=self.iterate)

  def iterate(self):
    while self.running:
      if not (isnan(self.price) or isnan(self.open_interest)):
        self.clear()
        self.calc()
        self.preserve()
        self.write()
      sleep(TIMEOUT)

  def clear(self):

    # Clear trades (within measurement window)
    now = datetime.utcnow()
    with LOCK:
      while len(self.trades) and now - self.trades[0]['timestamp'] > VF_MA_LEN_TD:
        self.trades.popleft()

    # Clear oldest history entries
    while len(self.buy_flow_history) >= VF_HIST_LEN:
      self.price_history.popleft()
      self.buy_flow_history.popleft()
      self.sell_flow_history.popleft()
      self.open_interest_history.popleft()

    # Clear recent trades
    while len(self.recent_trades) > RECENT_TRADES_LEN:
      self.recent_trades.pop()

  def _calc(self, side):
    with LOCK:
      return round(sum([trade['size'] for trade in self.trades
                                      if trade['side'] == side]) / VF_MA_LEN)

  def calc(self):
    self.buy_flow = self._calc(side='Buy')
    self.sell_flow = self._calc(side='Sell')

  def preserve(self):
    with LOCK:
      now = int(1000 * time())
      self.buy_flow_history.append({'time': now, 'buyFlow': self.buy_flow})
      self.sell_flow_history.append({'time': now, 'sellFlow': self.sell_flow})
      self.price_history.append({'time': now, 'price': self.price})
      self.open_interest_history.append({'time': now,
                                         'openInterest': self.open_interest})

  def write(self):
    with LOCK:
      for name, content in {
        PRICE_FILE: str(self.price),
        BUY_FLOW_FILE: str(self.buy_flow),
        SELL_FLOW_FILE: str(self.sell_flow),
        OPEN_INTEREST_FILE: str(self.open_interest),
        PRICE_HISTORY_FILE: dumps(list(self.price_history)),
        SELL_FLOW_HISTORY_FILE: dumps(list(self.sell_flow_history)),
        BUY_FLOW_HISTORY_FILE: dumps(list(self.buy_flow_history)),
        OPEN_INTEREST_HISTORY_FILE: dumps(list(self.open_interest_history)),
        OPEN_INTERESTS_FILE: dumps(self.open_interests),
        RECENT_TRADES_FILE: dumps(list(self.recent_trades), default=str),
      }.items():
        quick_write(name, content)

  def update(self, observable, message):

    # Measure price and volume
    if message['table'] == 'trade':
      if message['action'] == 'partial' or message['action'] == 'insert':
        with LOCK:
          timestamp = None
          side = None
          size = 0
          prices = []
          for datum in message['data']:
            if timestamp is None:
              timestamp = parse(datum['timestamp']).replace(tzinfo=None)
            side = datum['side']
            size += datum['size']
            self.price = datum['price']
            prices.append(self.price)
          price = round(mean(prices), 1)
          trade = {'timestamp': timestamp, 'side': side,
                   'size': size, 'price': price}
          if size > RECENT_TRADES_MIN_SIZE:
            self.recent_trades.appendleft(trade)
          self.trades.append(trade)

    # Measure open interest
    elif message['table'] == 'instrument':
      if message['action'] == 'partial' or message['action'] == 'update':
        data = message['data'][0]
        if 'openInterest' in data:

          # Extract open interest and compute total open interest
          self.open_interests[data['symbol']] = data['openInterest']
          oi = [oi for f, oi in self.open_interests.items()
                             if f in self.futures]
          self.open_interest = sum(oi)

# Putting everything together
if __name__ == '__main__':

  # Start polling daemons
  start_daemon(poll_fng)
  start_daemon(poll_cmc)
  start_daemon(poll_deribit)

  # Initiate a BitMEX client
  client = BitmexClient(key=None, secret=None)

  # Pick up the active bitcoin futures
  futures = [i['symbol'] for i in client.rest.active_instruments()
                         if i['rootSymbol'] == 'XBT'
                         and not i['symbol'].startswith('XBT7D')]

  # Configure websocket
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
  client.websocket.manual_subscriptions = [f'instrument:{i}' for i in futures]

  # Plug in volume flow measurer
  vfm = VolumeFlowMeasurer(futures)
  client.websocket.message_notifier.add_observer(vfm)
  client.websocket.close_notifier.add_observer(Stopper(vfm))

  # Plug in OHLC tracker
  # client.websocket.message_notifier.add_observer(OhlcTracker())

  # And bam
  client.connect()
