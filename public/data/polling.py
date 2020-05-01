from DeribitRest import DeribitRest
from quick_write import quick_write

from urllib.request import urlopen
from json import loads, dumps
from os.path import exists
from time import time, sleep
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects

from pprint import PrettyPrinter
pp = PrettyPrinter()

### Constants

# Fear and Greed
FNG_URL = 'https://api.alternative.me/fng/'
FNG_FILE = 'fear-and-greed.json'

# CoinMarketCap
BTCD_URL = 'https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest'
BTCD_FILE = 'bitcoin-dominance.txt'
TMC_FILE = 'total-market-cap.txt'
TV_FILE = 'total-volume.txt'
AV_FILE = 'altcoin-volume.txt'
AMC_FILE = 'altcoin-market-cap.txt'
CMC_BETWEEN_ITERS = 600# seconds (6 polls/hour or every 10 minutes)

# Deribit
CURRENCY = 'BTC'
HV_FILE = 'historical-volatility.txt'
DERIBIT_BETWEEN_ITERS = 1# seconds

### Polling functions

def poll_fng():

  # The actual requesting and writing the response to file
  def update():
    with urlopen(FNG_URL) as fng:
      quick_write(FNG_FILE, dumps(loads(fng.read())))

  # Dancing around edge cases
  try:
    while True:

      # Create local file if it doesn't exist yet
      if not exists(FNG_FILE):
        update()
        
      # Read our local file
      else:
        with open(FNG_FILE, 'r') as fngf:
          fng = loads(fngf.read())

          # Extract timestamp and update if necessary
          data = fng['data'][0]
          timestamp = int(data['timestamp'])
          until_update = int(data['time_until_update'])
          if time() - timestamp > until_update:
            update()

          # Kind of makeshift log message
          print(f'Time until Fear and Greed update: {until_update} seconds')
        sleep(until_update)

  except KeyboardInterrupt:
    print('Fear and Greed Index polling interrupted.')

def poll_cmc():

  # Setup
  headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': '522e4a07-85df-49e5-8b10-db5c64372920',
  }
  session = Session()
  session.headers.update(headers)

  try:
    while True:
      response = session.get(BTCD_URL)

      data = loads(response.text)['data']
      quick_write(BTCD_FILE, str(data['btc_dominance']))

      quote = data['quote']['USD']
      quick_write(TMC_FILE, str(quote['total_market_cap']))
      quick_write(TV_FILE, str(quote['total_volume_24h']))
      quick_write(AV_FILE, str(quote['altcoin_volume_24h']))
      quick_write(AMC_FILE, str(quote['altcoin_market_cap']))

      sleep(CMC_BETWEEN_ITERS)

  except (ConnectionError, Timeout, TooManyRedirects) as e:
    print(e)
  except KeyboardInterrupt:
    print('Bitcoin Dominance polling interrupted.')

def poll_deribit():
  try:
    rest = DeribitRest(auth=False)

    while True:
      hv = rest.historical_volatility('BTC')[-1][1]
      
      quick_write(HV_FILE, str(hv))
      
      sleep(DERIBIT_BETWEEN_ITERS)

  except KeyboardInterrupt:
    print('Deribit polling interrupted.')

if __name__ == '__main__':
  pass
