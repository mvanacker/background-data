const mongoose = require('mongoose');
const candleSchema = require('./schema.candle');
const Candle = mongoose.model('Candle', candleSchema);
module.exports = Candle;