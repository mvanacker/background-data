const mongoose = require('mongoose');
const Schema = mongoose.Schema;
const candleSchema = new Schema({
  symbol:          String,
  timeframe:       { type: String, required: true },
  timestamp:       Date,
  open:            Number,
  high:            Number,
  low:             Number,
  close:           Number,
  volume:          Number,

  // not actually used at time of writing
  trades:          Number,
  vwap:            Number,
  lastSize:        Number,
  turnover:        Number,
  homeNotional:    Number,
  foreignNotional: Number,

  // indicators
  sma_10:          Number,
  sma_200:         Number,

  ema_21:          Number,
  ema_55:          Number,
  ema_89:          Number,
  ema_200:         Number,
  ema_377:         Number,

  stoch:           Number,
  stoch_K:         Number,
  stoch_K_D:       Number,

  change_up:       Number,
  change_down:     Number,
  rma_up:          Number,
  rma_down:        Number,
  rsi:             Number,

  hv:              Number,
  hvp:             Number,
  hvp_ma:          Number,

  // forecast field(s)
  level:           Number,
}, {
  timestamps: true,
});
module.exports = candleSchema;