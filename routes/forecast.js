const router = require('express').Router();
const Forecast = require('../models/model.forecast');

router.route('/').get((req, res) => {
  const { timeframe } = req.query;
  Forecast.find({ timeframe }).sort({ level: 1, timestamp: 1 }).lean()
  .then(candles => res.json(candles))
  .catch(err => res.status(400).json(err));
});

router.route('/').delete((req, res) => {
  Forecast.deleteMany()
  .then(result => res.json(result))
  .catch(err => res.status(400).json(err));
})

router.route('/').put((req, res) => {
  // Note: a rethink may be necessary about how to specify timeframe and other
  // conditions (in general across candles, partials and forecasts).
  
  // A more efficient approach could be to only specify them in the query, and
  // then add them onto the object here.
  
  // A more robust approach could be to require them in both query and body and
  // then perform validation here to check whether one matches the other.

  // The most flexible approach would be to allow a condition in either query or
  // body, returning an error if it is not present in either. In the case that
  // is present in both, either document an order or return an error in case
  // they contradict.
  // (*) I like this last approach with an error in case of contradiction.

  const { timeframe, level } = req.query;
  const { timestamp } = req.body;
  const conditions = { timeframe, level, timestamp };
  Forecast.findOneAndUpdate(conditions, req.body, { upsert: true }).lean()
  .then(candle => res.json(candle))
  .catch(err => res.status(400).json(err));
});

module.exports = router;