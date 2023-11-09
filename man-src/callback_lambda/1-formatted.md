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
    ```python
    # Print the batch number at the beginning of every batch.
    batch_print_callback = LambdaCallback(
        on_train_batch_begin=lambda batch,logs: print(batch))

    # Stream the epoch loss to a file in JSON format. The file content
    # is not well-formed JSON but rather has a JSON object per line.
    import json
    json_log = open('loss_log.json', mode='wt', buffering=1)
    json_logging_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: json_log.write(
            json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '
'),
        on_train_end=lambda logs: json_log.close()
    )

    # Terminate some processes after having finished model training.
    processes = ...
    cleanup_callback = LambdaCallback(
        on_train_end=lambda logs: [
            p.terminate() for p in processes if p.is_alive()])

    model.fit(...,
              callbacks=[batch_print_callback,
                         json_logging_callback,
                         cleanup_callback])
    ```

@param on_epoch_begin called at the beginning of every epoch.
@param on_epoch_end called at the end of every epoch.
@param on_train_begin called at the beginning of model training.
@param on_train_end called at the end of model training.
@param on_train_batch_begin called at the beginning of every train batch.
@param on_train_batch_end called at the end of every train batch.
@param ... Any function in `Callback` that you want to override by
    passing `function_name=function`. For example,
    `LambdaCallback(.., on_train_end=train_end_fn)`. The custom function
    needs to have same arguments as the ones defined in `Callback`.

@export
@family callback
@seealso
+ <https:/keras.io/api/callbacks/lambda_callback#lambdacallback-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LambdaCallback>
