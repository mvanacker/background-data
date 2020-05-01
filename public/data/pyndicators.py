from collections import deque
from math import nan

class Indicator():
  def __init__(self, history=[]):
    self.history = history

  def __iter__(self):
    return iter(self.history)

  def tick(self, *args, **kwargs):
    self.history.append(self.partial(*args, **kwargs))

  def partial(self, *args, **kwargs):
    pass
    
class SMA(Indicator):
  def __init__(self, values, period, history=[]):
    super().__init__(history)

    self.window = deque(values[:period - 1])
    self.period = period
    self.sum = sum(self.window)

    # Fill history
    for value in values[period - 1:]:
      self.tick(value)

  def tick(self, value):
    super().tick(value)

    if len(self.window) == self.period - 1:
      self.sum -= self.window.popleft()
    self.sum += value
    self.window.append(value)

  def partial(self, value):
    if len(self.window) == self.period - 1:
      return (self.sum + value) / self.period
    return nan

class Stoch(Indicator):
  def __init__(self, values, highs, lows, period, history=[]):
    super().__init__(history)

    assert len(values) == len(lows)
    assert len(values) == len(highs)

    self.window = deque(values[-(period - 1):])
    self.window_highs = deque(highs[-(period - 1):])
    self.window_lows = deque(lows[-(period - 1):])
    self.period = period

    self.high = max(self.window_highs)
    self.low = min(self.window_lows)

  def tick(self, value, high, low):
    super().tick(value, high, low)

    if len(self.window) == self.period - 1:
      self.window.popleft()
      self.window_highs.popleft()
      self.window_lows.popleft()

    self.window.append(value)
    self.window_highs.append(high)
    self.window_lows.append(low)

    self.high = max(high, self.high)
    self.low = min(low, self.low)

  def partial(self, value, high, low):
    high = max(high, self.high)
    low = min(low, self.low)
    return (value - low) / (high - low)
