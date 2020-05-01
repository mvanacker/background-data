const router = require('express').Router();
const User = require('../models/user.model');

router.route('/').get((req, res) =>
  User.find()
  .then(users => res.json(users))
  .catch(err => res.status(400).json(err))
);

router.route('/add').post((req, res) => {
  new User(req.body).save()
  .then(() => res.json('User added.'))
  .catch(err => res.status(400).json(err))
});

module.exports = router;
