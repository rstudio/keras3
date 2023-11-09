Callback for creating simple, custom callbacks on-the-fly.

@description
This callback is constructed with anonymous functions that will be called
at the appropriate time (during `Model.{fit | evaluate | predict}`).
Note that the callbacks expects positional arguments, as:

- `on_epoch_begin` and `on_epoch_end` expect two positional arguments:
  `epoch`, `logs`
- `on_train_begin` and `on_train_end` expect one positional argument:
  `logs`
- `on_train_batch_begin` and `on_train_batch_end` expect two positional
  arguments: `batch`, `logs`
- See `Callback` class definition for the full list of functions and their
  expected arguments.

# Examples

```r
# Print the batch number at the beginning of every batch.
batch_print_callback <- callback_lambda(
  on_train_batch_begin = function(batch, logs) {
    print(batch)
  }
)

# Stream the epoch loss to a file in new-line delimited JSON format
# (one valid JSON object per line)
json_log <- file('loss_log.json', open = 'wt')
json_logging_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    jsonlite::write_json(
      list(epoch = epoch, loss = logs$loss),
      json_log,
      append = TRUE
    )
  },
  on_train_end = function(logs) {
    close(json_log)
  }
)

# Terminate some processes after having finished model training.
processes <- ...
cleanup_callback <- callback_lambda(
  on_train_end = function(logs) {
    for (p in processes) {
      if (is_alive(p)) {
        terminate(p)
      }
    }
  }
)

model %>% fit(
  ...,
  callbacks = list(
    batch_print_callback,
    json_logging_callback,
    cleanup_callback
  )
)
```

@param on_epoch_begin called at the beginning of every epoch.
@param on_epoch_end called at the end of every epoch.
@param on_train_begin called at the beginning of model training.
@param on_train_end called at the end of model training.
@param on_train_batch_begin called at the beginning of every train batch.
@param on_train_batch_end called at the end of every train batch.
@param ... Any function in `Callback` that you want to override by
    passing `function_name=function`. For example,
    `callback_lambda(.., on_train_end=train_end_fn)`. The custom function
    needs to have same arguments as the ones defined in `Callback`.

@export
@family callback
@seealso
+ <https:/keras.io/api/callbacks/lambda_callback#lambdacallback-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LambdaCallback>
