from collections import deque

class SMA():
  def __init__(self, df, src='close', period=10):
    self.name = f'sma-{period}'
    self.df = df
    
  def apply(self):
    pass

  def tick(self, row):
    pass

  def partial(self, row):
    pass

def sma(df, src='close', period=10):
  '''Add a simple moving average column to a dataframe.'''
  name = f'sma_{period}'

  # Only fill if indicator is already present
  if name in df:
    i = df[name].last_valid_index()
    l = df.index.get_loc(i)
    fill = df[src].iloc[l - period + 2:].rolling(period).mean()#.dropna()
    df[name].update(fill, overwrite=False)
    # Note: an alternative to .dropna() is to
    #       specify overwrite=False in .update()

  # Apply indicator to all rows
  else:
    df[name] = df[src].rolling(period).mean()