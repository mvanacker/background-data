const router = require('express').Router();
const Candle = require('../models/model.candle');

const COLUMNS = 'symbol timeframe timestamp open high low close volume sma_10 sma_200 ema_21 ema_55 ema_89 ema_200 ema_377 stoch stoch_K stoch_K_D change_up change_down rma_up rma_down rsi hv hvp';

router.route('/').get((req, res) => {
  const { timeframe, columns } = req.query;
  
  let { limit } = req.query;
  limit = parseInt(limit);

  selection = columns ? columns.split(',').join(' ') : COLUMNS;
  
  Candle.find({ timeframe }, selection)
  .limit(limit).sort({ timestamp: -1 })
  .then(candles => res.json(candles.reverse()))
  .catch(err => res.status(400).json(err));
});

router.route('/').post((req, res) => {
  new Candle(req.body).save()
  .then(candle => res.json(candle.toJSON()))
  .catch(err => res.status(400).json(err));
});

router.route('/batch').post((req, res) => {
  Candle.insertMany(req.body, {}, (err, docs) => {
    if (err) {
      res.json(err);
    } else {
      res.json(docs.length);
    }
  });
});

router.route('/count').get((req, res) => {
  const { timeframe } = req.query;
  Candle.countDocuments({ timeframe })
  .then(count => res.json(count))
  .catch(err => res.status(400).json(err));
});

router.route('/').put((req, res) => {
  const options = { new: true, upsert: true };
  Candle.findByIdAndUpdate(req.body._id, new Candle(req.body), options)
  .then(candle => res.json(candle))
  .catch(err => res.status(400).json(err));
});

router.route('/:id').patch((req, res) => {
  Candle.findByIdAndUpdate(req.params.id, req.body, { new: true })
  .then(candle => res.json(candle))
  .catch(err => res.status(400).json(err));
});

module.exports = router;