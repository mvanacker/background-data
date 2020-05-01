from collections import deque

class SMA():
  def __init__(self, df, src='close', period=10):
    self.name = f'sma-{period}'
    self.window = deque(df[src])

  def tick(self, row):
    pass

  def partial(self, row):
    pass