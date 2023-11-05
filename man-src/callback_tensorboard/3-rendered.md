Enable visualizations for TensorBoard.

@description
TensorBoard is a visualization tool provided with TensorFlow. A TensorFlow
installation is required to use this callback.

This callback logs events for TensorBoard, including:

* Metrics summary plots
* Training graph visualization
* Weight histograms
* Sampled profiling

When used in `model$evaluate()` or regular validation
in addition to epoch summaries, there will be a summary that records
evaluation metrics vs `model$optimizer$iterations` written. The metric names
will be prepended with `evaluation`, with `model$optimizer$iterations` being
the step in the visualized TensorBoard.

If you have installed TensorFlow with pip, you should be able
to launch TensorBoard from the command line:

```
tensorboard --logdir=path_to_your_logs
```

You can find more information about TensorBoard
[here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

# Examples
Basic usage:


```r
tensorboard_callback <- callback_tensorboard(log_dir = "./logs")
model %>% fit(x_train, y_train, epochs = 2, callbacks = list(tensorboard_callback))
```

```
## Error in eval(expr, envir, enclos): object 'model' not found
```

```r
# Then run the tensorboard command to view the visualizations.
```

Custom batch-level summaries in a subclassed Model:


```r
MyModel <- keras_model_custom(name = "MyModel", function(self) {
  self$dense <- layer_dense(units = 10)
  function (self, x, mask = NULL) {
    outputs <- self$dense(x)
    tf$summary$histogram('outputs', outputs)
    outputs
  }
})

model <- MyModel()
```

```
## RModel.call() missing 1 required positional argument: 'inputs'
```

```r
model %>% compile(optimizer = 'sgd', loss = 'mse')
```

```
## Error in eval(expr, envir, enclos): object 'model' not found
```

```r
# Make sure to set `update_freq=N` to log a batch-level summary every N
# batches.  In addition to any `tf.summary` contained in `model.call()`,
# metrics added in `Model.compile` will be logged every N batches.
tb_callback <- callback_tensorboard(log_dir = './logs', update_freq = 1)
model %>% fit(x_train, y_train, callbacks = list(tb_callback))
```

```
## Error in eval(expr, envir, enclos): object 'model' not found
```

Custom batch-level summaries in a Functional API Model:


```r
my_summary <- function(x) {
  tf$summary$histogram('x', x)
  x
}

inputs <- layer_input(shape = 10)
x <- layer_dense(units = 10)(inputs)
outputs <- layer_lambda(my_summary)(x)
```

```
## Lambda.__init__() missing 1 required positional argument: 'function'
```

```r
model <- keras_model(inputs = inputs, outputs = outputs)
```

```
## Error in eval(expr, envir, enclos): object 'outputs' not found
```

```r
model %>% compile(optimizer = 'sgd', loss = 'mse')
```

```
## Error in eval(expr, envir, enclos): object 'model' not found
```

```r
# Make sure to set `update_freq=N` to log a batch-level summary every N
# batches. In addition to any `tf.summary` contained in `Model.call`,
# metrics added in `Model.compile` will be logged every N batches.
tb_callback <- callback_tensorboard(log_dir = './logs', update_freq = 1)
model %>% fit(x_train, y_train, callbacks = list(tb_callback))
```

```
## Error in eval(expr, envir, enclos): object 'model' not found
```

Profiling:


```r
# Profile a single batch, e.g. the 5th batch.
tensorboard_callback <- callback_tensorboard(
  log_dir = './logs', profile_batch = 5)
model %>% fit(x_train, y_train, epochs = 2, callbacks = list(tensorboard_callback))
```

```
## Error in eval(expr, envir, enclos): object 'model' not found
```

```r
# Profile a range of batches, e.g. from 10 to 20.
tensorboard_callback <- callback_tensorboard(
  log_dir = './logs', profile_batch = c(10, 20))
model %>% fit(x_train, y_train, epochs = 2, callbacks = list(tensorboard_callback))
```

```
## Error in eval(expr, envir, enclos): object 'model' not found
```

@param log_dir the path of the directory where to save the log files to be
    parsed by TensorBoard. e.g.,
    `log_dir = file.path(working_dir, 'logs')`.
    This directory should not be reused by any other callbacks.
@param histogram_freq frequency (in epochs) at which to compute
    weight histograms for the layers of the model. If set to 0,
    histograms won't be computed. Validation data (or split) must be
    specified for histogram visualizations.
@param write_graph (Not supported at this time)
    Whether to visualize the graph in TensorBoard.
    Note that the log file can become quite large
    when `write_graph` is set to `TRUE`.
@param write_images whether to write model weights to visualize as image in
    TensorBoard.
@param write_steps_per_second whether to log the training steps per second
    into TensorBoard. This supports both epoch and batch frequency
    logging.
@param update_freq `"batch"` or `"epoch"` or integer. When using `"epoch"`,
    writes the losses and metrics to TensorBoard after every epoch.
    If using an integer, let's say `1000`, all metrics and losses
    (including custom ones added by `Model.compile`) will be logged to
    TensorBoard every 1000 batches. `"batch"` is a synonym for 1,
    meaning that they will be written every batch.
    Note however that writing too frequently to TensorBoard can slow
    down your training, especially when used with distribution
    strategies as it will incur additional synchronization overhead.
    Batch-level summary writing is also available via `train_step`
    override. Please see
    [TensorBoard Scalars tutorial](
        https://www.tensorflow.org/tensorboard/scalars_and_keras#batch-level_logging)  # noqa: E501
    for more details.
@param profile_batch (Not supported at this time)
    Profile the batch(es) to sample compute characteristics.
    profile_batch must be a non-negative integer or a tuple of integers.
    A pair of positive integers signify a range of batches to profile.
    By default, profiling is disabled.
@param embeddings_freq frequency (in epochs) at which embedding layers will be
    visualized. If set to 0, embeddings won't be visualized.
@param embeddings_metadata Dictionary which maps embedding layer names to the
    filename of a file in which to save metadata for the embedding layer.
    In case the same metadata file is to be
    used for all embedding layers, a single filename can be passed.

@export
@family callback
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard>
