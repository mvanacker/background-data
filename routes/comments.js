const router = require('express').Router();
const Comment = require('../models/comment.model');

router.route('/').get((req, res) =>
  Comment.find()
  .then(comments => res.json(comments))
  .catch(err => res.status(400).json(err))
);

router.route('/:id').get((req, res) =>
  Comment.findById(req.params.id)
  .then(comment => res.json(comment))
  .catch(err => res.status(400).json(err))
);

router.route('/add').post((req, res) => {
  new Comment(req.body).save()
  .then(result => res.json(result.toJSON()))
  .catch(err => res.status(400).json(err))
});

module.exports = router;