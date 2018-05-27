from basic_model import BasicModel
import tensorflow as tf

class BiDirLSTM(BasicModel):
    """Abstracts a Tensorflow graph for a learning task.
    """
    def _add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.
        """
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=())
        self.labels_placeholder = tf.placeholder(tf.int32, shape())
        self.dropout_placeholder = tf.placeholder(tf.float32, shape = ())

    def _create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for one step of training.
        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {self.inputs_placeholder: inputs_batch}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict
    def _add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        #with tf.variable_scope("RNN", reuse = tf.AUTO_REUSE):
        embeddings = tf.get_variable("embeddings", initializer = self.pretrained_embeddings,trainable=False)
        embeddings = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
        embeddings = tf.reshape(embeddings, [-1, self.max_length, self.config.n_features* self.config.embed_size])
        embeddings = tf.cast(embeddings, tf.float32)
        return embeddings        

    def _add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """

        model = tf.nn.bidirectional_dynamic_rnn()

        return pred

    def _add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def _add_training_op(self, loss):
        """Sets up the training Ops.
        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.
        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data
        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
