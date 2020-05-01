from quick_write import quick_read
from json import loads
from dateutil.parser import isoparse
from datetime import timedelta
raw_series = loads(quick_read('ohlc-1h.json'))
print(f'len: {len(raw_series)}')
p = lambda c: isoparse(c['timestamp'])
h = timedelta(minutes=60)
for i, (c1, c2) in enumerate(zip(raw_series[:-1], raw_series[1:])):
  if p(c2) - p(c1) != h:
    print(i)
    print(c1)
    print(c2)