const router = require('express').Router();
const Candle = require('../models/model.candle');

router.route('/').get((req, res) => {
  const { timeframe } = req.query;
  Candle.find({ timeframe }).sort({ timestamp: 1 })
  .then(candle => res.json(candle))
  .catch(err => res.status(400).json(err))
});

router.route('/add').post((req, res) => {
  new Candle(req.body).save()
  .then(candle => res.json(candle.toJSON()))
  .catch(err => res.status(400).json(err))
});

router.route('/count').get((req, res) => {
  const { timeframe } = req.query;
  Candle.countDocuments({ timeframe })
  .then(count => res.json(count))
  .catch(err => res.status(400).json(err))
});

router.route('/edit').put((req, res) => {
  const options = { new: true, upsert: true };
  Candle.findByIdAndUpdate(req.body._id, new Candle(req.body), options)
  .then(candle => res.json(candle))
  .catch(err => res.status(400).json(err))
});

router.route('/edit/:id').patch((req, res) => {
  Candle.findByIdAndUpdate(req.params.id, req.body, { new: true })
  .then(candle => res.json(candle))
  .catch(err => res.status(400).json(err))
});

module.exports = router;