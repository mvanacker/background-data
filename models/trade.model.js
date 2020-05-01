const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const tradeSchema = new Schema({
  exchange: {
    type: String,
    required: true,
    trim: true,
    minlength: 1,
  },
  instrument: {
    type: String,
    required: true,
    trim: true,
    minlength: 1,
  },
  entry: [{
    type: Number,
    required: true,
  }],
  stop: [{
    type: Number,
    required: true,
  }],
  profit: [{
    type: Number,
    required: true,
  }],
  risk: {
    type: Number,
    required: true,
  },
  quantity: {
    type: Number,
    required: true,
  },
  position: {
    type: String,
    trim: true,
    required: true,
  },
  contract: {
    type: String,
    trim: true,
    required: true,
  },
  comments: [{
    type: Schema.Types.ObjectId,
    ref: 'Comment',
  }],
}, {
  timestamps: true,
});

// tradeSchema.virtual('lastCommentCreatedAt').get(() => {
//   const creationDates = this.comments.map(comment =>
//     new Date(comment.createdAt)
//   );
//   return new Date(Math.max.apply(Math, creationDates));
// });

const Trade = mongoose.model('Trade', tradeSchema);

module.exports = Trade;
