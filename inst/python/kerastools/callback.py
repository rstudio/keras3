
import os

if (os.getenv('KERAS_IMPLEMENTATION', 'tensorflow') == 'keras'):
  from keras.callbacks import Callback
else:
  from tensorflow.python.keras.callbacks import Callback

class RCallback(Callback):

  def __init__(self, r_set_context,
                     r_on_epoch_begin,
                     r_on_epoch_end,
                     r_on_train_begin,
                     r_on_train_end,
                     r_on_batch_begin,
                     r_on_batch_end,
                     r_on_predict_batch_begin,
                     r_on_predict_batch_end,
                     r_on_predict_begin,
                     r_on_predict_end,
                     r_on_test_batch_begin,
                     r_on_test_batch_end,
                     r_on_test_begin,
                     r_on_test_end,
                     r_on_train_batch_begin,
                     r_on_train_batch_end
                     ):
    super(Callback, self).__init__()
    self.r_set_context = r_set_context
    self.r_on_epoch_begin = r_on_epoch_begin
    self.r_on_epoch_end = r_on_epoch_end
    self.r_on_train_begin = r_on_train_begin
    self.r_on_train_end = r_on_train_end
    self.r_on_batch_begin = r_on_batch_begin
    self.r_on_batch_end = r_on_batch_end
    self.r_on_predict_batch_begin = r_on_predict_batch_begin
    self.r_on_predict_batch_end = r_on_predict_batch_end
    self.r_on_predict_begin = r_on_predict_begin
    self.r_on_predict_end = r_on_predict_end
    self.r_on_test_batch_begin = r_on_test_batch_begin
    self.r_on_test_batch_end = r_on_test_batch_end
    self.r_on_test_begin = r_on_test_begin
    self.r_on_test_end = r_on_test_end
    self.r_on_train_batch_begin = r_on_train_batch_begin
    self.r_on_train_batch_end = r_on_train_batch_end

    # required when using tf$distributed strategies
    self._chief_worker_only = False

  def on_epoch_begin(self, epoch, logs=None):
    out = self.r_on_epoch_begin(epoch, logs)
    if isinstance(out, dict) and isinstance(logs, dict):
      logs.update(out)

  def on_epoch_end(self, epoch, logs=None):
    out = self.r_on_epoch_end(epoch, logs)
    if isinstance(out, dict) and isinstance(logs, dict):
      logs.update(out)

  def on_train_begin(self, logs=None):
    self.r_set_context(self.params, self.model)
    out = self.r_on_train_begin(logs)
    if isinstance(out, dict) and isinstance(logs, dict):
      logs.update(out)

  def on_train_end(self, logs=None):
    out = self.r_on_train_end(logs)
    if isinstance(out, dict) and isinstance(logs, dict):
      logs.update(out)

  def on_batch_begin(self, batch, logs=None):
    out = self.r_on_batch_begin(batch, logs)
    if isinstance(out, dict) and isinstance(logs, dict):
      logs.update(out)

  def on_batch_end(self, batch, logs=None):
    out = self.r_on_batch_end(batch, logs)
    if isinstance(out, dict) and isinstance(logs, dict):
      logs.update(out)

  def on_predict_batch_begin(self, batch, logs=None):
    out = self.r_on_predict_batch_begin(batch, logs)
    if isinstance(out, dict) and isinstance(logs, dict):
      logs.update(out)

  def on_predict_batch_end(self, batch, logs=None):
    out = self.r_on_predict_batch_end(batch, logs)
    if isinstance(out, dict) and isinstance(logs, dict):
      logs.update(out)

  def on_predict_begin(self, logs=None):
    out = self.r_on_predict_begin(logs)
    if isinstance(out, dict) and isinstance(logs, dict):
      logs.update(out)

  def on_predict_end(self, logs=None):
    out = self.r_on_predict_end(logs)
    if isinstance(out, dict) and isinstance(logs, dict):
      logs.update(out)

  def on_test_batch_begin(self, batch, logs=None):
    out = self.r_on_test_batch_begin(batch, logs)
    if isinstance(out, dict) and isinstance(logs, dict):
      logs.update(out)

  def on_test_batch_end(self, batch, logs=None):
    out = self.r_on_test_batch_end(batch, logs)
    if isinstance(out, dict) and isinstance(logs, dict):
      logs.update(out)

  def on_test_begin(self, logs=None):
    out = self.r_on_test_begin(logs)
    if isinstance(out, dict) and isinstance(logs, dict):
      logs.update(out)

  def on_test_end(self, logs=None):
    out = self.r_on_test_end(logs)
    if isinstance(out, dict) and isinstance(logs, dict):
      logs.update(out)

  def on_train_batch_begin(self, batch, logs=None):
    out = self.r_on_train_batch_begin(batch, logs)
    if isinstance(out, dict) and isinstance(logs, dict):
      logs.update(out)

  def on_train_batch_end(self, batch, logs=None):
    out = self.r_on_train_batch_end(batch, logs)
    if isinstance(out, dict) and isinstance(logs, dict):
      logs.update(out)
