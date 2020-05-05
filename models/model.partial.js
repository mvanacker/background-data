const mongoose = require('mongoose');
const candleSchema = require('./schema.candle');
const Partial = mongoose.model('Partial', candleSchema);
module.exports = Partial;