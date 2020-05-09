from math import exp, log, sqrt
from scipy.stats import norm

class PricingModel:
  '''Reminder to keep everything as general as possible.'''

  def premium_call(self, price, strike, volatility, time):
    pass

  def premium_put(self, price, strike, volatility, time):
    pass

  def delta_call(self, price, strike, volatility, time):
    pass

  def delta_put(self, price, strike, volatility, time):
    pass

  def gamma(self, price, strike, volatility, time):
    pass

  def vega(self, price, strike, volatility, time):
    pass

  def theta(self, price, strike, volatility, time):
    pass

  def prob_itm_call(self, price, strike, volatility, time):
    pass

  def prob_itm_put(self, price, strike, volatility, time):
    pass

  def prob_touch_call(self, price, strike, volatility, time):
    pass

  def prob_touch_put(self, price, strike, volatility, time):
    pass

  def prob_profit_call(self, price, strike, volatility, time):
    pass

  def prob_profit_put(self, price, strike, volatility, time):
    pass

class BlackScholes(PricingModel):
  '''Simplified Black-Scholes pricing model for use with currency (e.g. Bitcoin). Interest rate and dividend yield are implicitly set to 0 in the computations.'''

  def __init__(self):
    pass

  def _d(self, price, strike, volatility, time):
    vst = volatility * sqrt(time)
    d1 = (log(price / strike) + time * volatility ** 2 / 2) / vst
    d2 = d1 - vst
    return d1, d2

  def premium_call(self, price, strike, volatility, time):
    d1, d2 = self._d(price, strike, volatility, time)
    return price * norm.cdf(d1) - strike * norm.cdf(d2)

  def premium_put(self, price, strike, volatility, time):
    d1, d2 = self._d(price, strike, volatility, time)
    return strike * norm.cdf(-d2) - price * norm.cdf(-d1)

  def delta_call(self, price, strike, volatility, time):
    d1, _ = self._d(price, strike, volatility, time)
    return norm.cdf(d1)

  def delta_put(self, price, strike, volatility, time):
    d1, _ = self._d(price, strike, volatility, time)
    return norm.cdf(-d1)

  def gamma(self, price, strike, volatility, time):
    d1, _ = self._d(price, strike, volatility, time)
    return norm.pdf(d1) / (price * volatility * sqrt(time))

  def vega(self, price, strike, volatility, time):
    d1, _ = self._d(price, strike, volatility, time)
    return price * sqrt(time) * norm.pdf(d1)

  def theta(self, price, strike, volatility, time):
    d1, _ = self._d(price, strike, volatility, time)
    return price * norm.pdf(d1) * volatility / (2 * sqrt(time))

  def prob_itm_call(self, price, strike, volatility, time):
    _, d2 = self._d(price, strike, volatility, time)
    return norm.cdf(d2)

  def prob_itm_put(self, price, strike, volatility, time):
    _, d2 = self._d(price, strike, volatility, time)
    return norm.cdf(-d2)

  def prob_touch_call(self, price, strike, volatility, time):
    _, d2 = self._d(price, strike, volatility, time)
    prob = norm.cdf(d2)
    return 2 * (prob if prob < 0.5 else 1 - prob)

  def prob_touch_put(self, price, strike, volatility, time):
    _, d2 = self._d(price, strike, volatility, time)
    prob = norm.cdf(-d2)
    return 2 * (prob if prob < 0.5 else 1 - prob)

  def prob_profit_call(self, price, strike, volatility, time):
    prem = self.premium_call(price, strike, volatility, time)
    return self.prob_itm_call(price, strike + prem, volatility, time)
 
  def prob_profit_put(self, price, strike, volatility, time):
    prem = self.premium_put(price, strike, volatility, time)
    return self.prob_itm_put(price, strike - prem, volatility, time)

  def reverse(self, prob, strike, volatility, time):
    vt = volatility ** 2 * time / 2
    svt = sqrt(vt)
    foo = norm.ppf(prob) + svt
    lower = strike * exp(foo * svt - vt)
    upper = strike * exp(-foo * svt - vt)
    return lower, upper
