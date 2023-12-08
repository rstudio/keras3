from keras.callbacks import Callback

from functools import wraps


def wrap_sig_idx_logs(fn):
    # print("Calling wrap_sig_idx_logs")
    @wraps(fn)
    def wrapper(self, idx, logs=None):
        # print(f"In wrapper {idx=}  {logs=}")
        res = fn(self, idx+1, logs)
        # print(f"{res=}")
        if isinstance(res, dict):
            # print("updating logs")
            logs.update(res)
    return wrapper


def wrap_sig_logs(fn):
    @wraps(fn)
    def wrapper(self, logs=None):
        res = fn(self, logs)
        if isinstance(res, dict):
            logs.update(res)
    return wrapper



class RCallback(Callback):
    def __init__(
        self,
        r_set_context,
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
        r_on_train_batch_end,
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
        self.r_on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.r_on_epoch_end(epoch, logs)

    def on_train_begin(self, logs=None):
        self.r_set_context(self.params, self.model)
        self.r_on_train_begin(logs)

    def on_train_end(self, logs=None):
        self.r_on_train_end(logs)

    def on_batch_begin(self, batch, logs=None):
        self.r_on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        self.r_on_batch_end(batch, logs)

    def on_predict_batch_begin(self, batch, logs=None):
        self.r_on_predict_batch_begin(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        self.r_on_predict_batch_end(batch, logs)

    def on_predict_begin(self, logs=None):
        self.r_on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        self.r_on_predict_end(logs)

    def on_test_batch_begin(self, batch, logs=None):
        self.r_on_test_batch_begin(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        self.r_on_test_batch_end(batch, logs)

    def on_test_begin(self, logs=None):
        self.r_on_test_begin(logs)

    def on_test_end(self, logs=None):
        self.r_on_test_end(logs)

    def on_train_batch_begin(self, batch, logs=None):
        self.r_on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        self.r_on_train_batch_end(batch, logs)
