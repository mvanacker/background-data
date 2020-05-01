const router = require('express').Router();
const Trade = require('../models/trade.model');

router.route('/').get((req, res) =>
  Trade.aggregate([{
    $lookup: {
      from: 'comments',
      localField: 'comments',
      foreignField: '_id',
      as: 'comments'
    }
  }, {
    $addFields: {
      latestCommentCreatedAt: {
        $max: "$comments.createdAt"
      }
    }
  }, {
    $sort: {
      latestCommentCreatedAt: -1
    }
  }])
  .then(trades => res.json(trades))
  .catch(err => res.status(400).json(err))
);

router.route('/:id').get((req, res) =>
  Trade.findById(req.params.id)
  .then(trade => res.json(trade))
  .catch(err => res.status(400).json(err))
);

router.route('/add').post((req, res) =>
  new Trade(req.body).save()
  .then(trade => res.json(trade.toJSON()))
  .catch(err => res.status(400).json(err))
);

router.route('/update/:id').put((req, res) =>
  Trade.findOneAndUpdate({
    _id: req.params.id
  }, new Trade(req.body)).exec()
  .then(trade => res.json(trade.toJSON()))
  .catch(err => res.status(400).json(err))
);

router.route('/delete/empty').delete((req, res) =>
  Trade.deleteMany({
    comments: {
      $size: 0,
    },
  })
  .then(trades => res.json(trades))
  .catch(err => res.status(400).json(err))
);

module.exports = router;
