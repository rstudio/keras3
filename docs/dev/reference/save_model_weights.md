# Saves all weights to a single file or sharded files.

By default, the weights are saved in a single `.weights.h5` file. Enable
sharding via `max_shard_size` to split weights across multiple files (in
GB) and produce a `.weights.json` manifest that tracks shard metadata.

The saved sharded files contain:

- `*.weights.json`: configuration file containing `metadata` and
  `weight_map` entries.

- `*_xxxxxx.weights.h5`: weight shards limited by `max_shard_size`.

    model <-
      keras_model_sequential(input_shape = 2) |>
      layer_dense(4)

    path_h5 <- tempfile(fileext = ".weights.h5")
    path_json <- tempfile(fileext = ".weights.json")

    model |> save_model_weights(path_h5)
    model |> save_model_weights(path_json, max_shard_size = 0.01)

    model |> load_model_weights(path_h5)
    model |> load_model_weights(path_json)

## Usage

``` r
save_model_weights(model, filepath, overwrite = FALSE, max_shard_size = NULL)
```

## Arguments

- model:

  A keras Model object.

- filepath:

  Path where the weights will be saved. Accepts `.weights.h5`, or when
  sharding is enabled, a `.weights.json` manifest path. If `.weights.h5`
  is provided while sharding, the filename will be overridden to end in
  `.weights.json`.

- overwrite:

  Whether to overwrite any existing weights at the target location, or
  instead ask the user via an interactive prompt.

- max_shard_size:

  Numeric size in GB for each sharded file. Use `NULL` to disable
  sharding.

## Value

This is called primarily for side effects. `model` is returned,
invisibly, to enable usage with the pipe.

## See also

- <https://keras.io/api/models/model_saving_apis/weights_saving_and_loading#saveweights-method>

Other saving and loading functions:  
[`export_savedmodel.keras.src.models.model.Model()`](https://keras3.posit.co/dev/reference/export_savedmodel.keras.src.models.model.Model.md)  
[`layer_tfsm()`](https://keras3.posit.co/dev/reference/layer_tfsm.md)  
[`load_model()`](https://keras3.posit.co/dev/reference/load_model.md)  
[`load_model_weights()`](https://keras3.posit.co/dev/reference/load_model_weights.md)  
[`register_keras_serializable()`](https://keras3.posit.co/dev/reference/register_keras_serializable.md)  
[`save_model()`](https://keras3.posit.co/dev/reference/save_model.md)  
[`save_model_config()`](https://keras3.posit.co/dev/reference/save_model_config.md)  
[`with_custom_object_scope()`](https://keras3.posit.co/dev/reference/with_custom_object_scope.md)  
