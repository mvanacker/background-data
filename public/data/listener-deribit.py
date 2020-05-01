from DeribitWebsocket import DeribitWebsocket
from DeribitRest import DeribitRest
from AObserver import AObserver
from quick_write import quick_write

from json import dumps

# Testing stuff
# import asyncio
# from time import sleep
from pprint import PrettyPrinter
pp = PrettyPrinter()

INSTRUMENTS_FILE = 'instruments.json'

# Init rest client
rest = DeribitRest(auth=False)

# Get all futures contract names
instruments = [data['instrument_name'] for data in rest.public_api.public_get_instruments_get('BTC', kind='future')['result']]
quick_write(INSTRUMENTS_FILE, dumps(instruments))

# Derive instrument's quote channels
futures_channels = []#[f'quote.{instrument}' for instrument in instruments]

# All public channels
public_channels = futures_channels

# All private channels
private_channels = [f'user.portfolio.BTC']

class Writer(AObserver):
  def __init__(self):
    pass

  async def update(self, observable, message):
    if 'params' in message and 'channel' in message['params'] and 'data' in message['params']:
      channel = message['params']['channel']
      quick_write(f'{channel}.json', dumps(message['params']['data']))
    else:
      print('--- (!) Uncaught message:')
      pp.pprint(message)

# Initialize socket
socket = DeribitWebsocket(test=False)

# Register channels
for channel in public_channels:
  socket.public_channels.append(channel)
for channel in private_channels:
  socket.private_channels.append(channel)

# Register listener and go
socket.message_notifier.add_observer(Writer())
socket.connect(auto_reconnect=True)
