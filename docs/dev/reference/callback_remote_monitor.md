# Callback used to stream events to a server.

Requires the `requests` library. Events are sent to
`root + '/publish/epoch/end/'` by default. Calls are HTTP POST, with a
`data` argument which is a JSON-encoded named list of event data. If
`send_as_json = TRUE`, the content type of the request will be
`"application/json"`. Otherwise the serialized JSON will be sent within
a form.

## Usage

``` r
callback_remote_monitor(
  root = "http://localhost:9000",
  path = "/publish/epoch/end/",
  field = "data",
  headers = NULL,
  send_as_json = FALSE
)
```

## Arguments

- root:

  String; root url of the target server.

- path:

  String; path relative to `root` to which the events will be sent.

- field:

  String; JSON field under which the data will be stored. The field is
  used only if the payload is sent within a form (i.e. when
  `send_as_json = FALSE`).

- headers:

  Named list; optional custom HTTP headers.

- send_as_json:

  Boolean; whether the request should be sent as `"application/json"`.

## Value

A `Callback` instance that can be passed to
[`fit.keras.src.models.model.Model()`](https://keras3.posit.co/dev/reference/fit.keras.src.models.model.Model.md).

## See also

- <https://keras.io/api/callbacks/remote_monitor#remotemonitor-class>

Other callbacks:  
[`Callback()`](https://keras3.posit.co/dev/reference/Callback.md)  
[`callback_backup_and_restore()`](https://keras3.posit.co/dev/reference/callback_backup_and_restore.md)  
[`callback_csv_logger()`](https://keras3.posit.co/dev/reference/callback_csv_logger.md)  
[`callback_early_stopping()`](https://keras3.posit.co/dev/reference/callback_early_stopping.md)  
[`callback_lambda()`](https://keras3.posit.co/dev/reference/callback_lambda.md)  
[`callback_learning_rate_scheduler()`](https://keras3.posit.co/dev/reference/callback_learning_rate_scheduler.md)  
[`callback_model_checkpoint()`](https://keras3.posit.co/dev/reference/callback_model_checkpoint.md)  
[`callback_reduce_lr_on_plateau()`](https://keras3.posit.co/dev/reference/callback_reduce_lr_on_plateau.md)  
[`callback_swap_ema_weights()`](https://keras3.posit.co/dev/reference/callback_swap_ema_weights.md)  
[`callback_tensorboard()`](https://keras3.posit.co/dev/reference/callback_tensorboard.md)  
[`callback_terminate_on_nan()`](https://keras3.posit.co/dev/reference/callback_terminate_on_nan.md)  
