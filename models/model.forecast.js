const mongoose = require('mongoose');
const forecastSchema = require('./schema.candle');
const Forecast = mongoose.model('Forecast', forecastSchema);
module.exports = Forecast;