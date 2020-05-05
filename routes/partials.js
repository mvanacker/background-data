const router = require('express').Router();
const Partial = require('../models/model.partial');

router.route('/').get((req, res) => {
  const { timeframe } = req.query;
  Partial.findOne({ timeframe })
  .then(candles => res.json(candles))
  .catch(err => res.status(400).json(err));
});

router.route('/').put((req, res) => {
  const { timeframe } = req.query;
  Partial.findOneAndUpdate({ timeframe }, req.body, { upsert: true })
  .then(candle => res.json(candle))
  .catch(err => res.status(400).json(err));
});

module.exports = router;