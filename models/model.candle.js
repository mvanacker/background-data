const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const Candle = mongoose.model('Candle', new Schema({
  symbol:          String,
  timeframe:       { type: String, required: true },
  timestamp:       Date,
  open:            Number,
  high:            Number,
  low:             Number,
  close:           Number,
  trades:          Number,
  volume:          Number,
  vwap:            Number,
  lastSize:        Number,
  turnover:        Number,
  homeNotional:    Number,
  foreignNotional: Number,

  // indicators
  sma_10:          Number,
  stochK:          Number,
  stochD:          Number,
  rsi:             Number,
  rsi_ema:         Number,
  hvp:             Number,
}, {
  timestamps: true,
}));

module.exports = Candle;