from quick_write import quick_read

from json import loads
from dateutil.parser import isoparse
from datetime import datetime
import pandas as pd

raw_series = loads(quick_read(f'ohlc-1h.json'))
df = pd.DataFrame(raw_series)
df['timestamp'] = df['timestamp'].apply(isoparse)
df = df.set_index('timestamp')
print(df.tail())
