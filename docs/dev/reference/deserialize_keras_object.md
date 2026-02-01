# Retrieve the object by deserializing the config dict.

The config dict is a Python dictionary that consists of a set of
key-value pairs, and represents a Keras object, such as an `Optimizer`,
`Layer`, `Metrics`, etc. The saving and loading library uses the
following keys to record information of a Keras object:

- `class_name`: String. This is the name of the class, as exactly
  defined in the source code, such as "LossesContainer".

- `config`: Named List. Library-defined or user-defined key-value pairs
  that store the configuration of the object, as obtained by
  `object$get_config()`.

- `module`: String. The path of the python module. Built-in Keras
  classes expect to have prefix `keras`.

- `registered_name`: String. The key the class is registered under via
  `register_keras_serializable(package, name)` API. The key has the
  format of `'{package}>{name}'`, where `package` and `name` are the
  arguments passed to
  [`register_keras_serializable()`](https://keras3.posit.co/dev/reference/register_keras_serializable.md).
  If `name` is not provided, it uses the class name. If
  `registered_name` successfully resolves to a class (that was
  registered), the `class_name` and `config` values in the config dict
  will not be used. `registered_name` is only used for non-built-in
  classes.

For example, the following config list represents the built-in Adam
optimizer with the relevant config:

    config <- list(
      class_name = "Adam",
      config = list(
        amsgrad = FALSE,
        beta_1 = 0.8999999761581421,
        beta_2 = 0.9990000128746033,
        epsilon = 1e-07,
        learning_rate = 0.0010000000474974513,
        name = "Adam"
      ),
      module = "keras.optimizers",
      registered_name = NULL
    )
    # Returns an `Adam` instance identical to the original one.
    deserialize_keras_object(config)

    ## <keras.src.optimizers.adam.Adam object at 0x0>

If the class does not have an exported Keras namespace, the library
tracks it by its `module` and `class_name`. For example:

    config <- list(
      class_name = "MetricsList",
      config =  list(
        ...
      ),
      module = "keras.trainers.compile_utils",
      registered_name = "MetricsList"
    )

    # Returns a `MetricsList` instance identical to the original one.
    deserialize_keras_object(config)

And the following config represents a user-customized `MeanSquaredError`
loss:

    # define a custom object
    loss_modified_mse <- Loss(
      "ModifiedMeanSquaredError",
      inherit = loss_mean_squared_error)

    # register the custom object
    register_keras_serializable(loss_modified_mse)

    ## function (reduction = "sum_over_batch_size", name = "mean_squared_error",
    ##     dtype = NULL)
    ## {
    ##     args <- capture_args(enforce_all_dots_named = FALSE)
    ##     do.call(ModifiedMeanSquaredError, args)
    ## }
    ## <environment: 0x5dddb2eef3e0>

    # confirm object is registered
    get_custom_objects()

    ## $`keras3>ModifiedMeanSquaredError`
    ## <class '<r-namespace:keras3>.ModifiedMeanSquaredError'>
    ##  signature: (
    ##    reduction='sum_over_batch_size',
    ##    name='mean_squared_error',
    ##    dtype=None
    ## )

    get_registered_name(loss_modified_mse)

    ## [1] "keras3>ModifiedMeanSquaredError"

    # now custom object instances can be serialized
    full_config <- serialize_keras_object(loss_modified_mse())

    # the `config` arguments will be passed to loss_modified_mse()
    str(full_config)

    ## List of 4
    ##  $ module         : chr "<r-namespace:keras3>"
    ##  $ class_name     : chr "ModifiedMeanSquaredError"
    ##  $ config         :List of 2
    ##   ..$ name     : chr "mean_squared_error"
    ##   ..$ reduction: chr "sum_over_batch_size"
    ##  $ registered_name: chr "keras3>ModifiedMeanSquaredError"

    # and custom object instances can be deserialized
    deserialize_keras_object(full_config)

    ## <LossFunctionWrapper(<function mean_squared_error at 0x0>, kwargs={})>
    ##  signature: (y_true, y_pred, sample_weight=None)

    # Returns the `ModifiedMeanSquaredError` object

## Usage

``` r
deserialize_keras_object(config, custom_objects = NULL, safe_mode = TRUE, ...)
```

## Arguments

- config:

  Named list describing the object.

- custom_objects:

  Named list containing a mapping between custom object names the
  corresponding classes or functions.

- safe_mode:

  Boolean, whether to disallow unsafe `lambda` deserialization. When
  `safe_mode=FALSE`, loading an object has the potential to trigger
  arbitrary code execution. This argument is only applicable to the
  Keras v3 model format. Defaults to `TRUE`.

- ...:

  For forward/backward compatability.

## Value

The object described by the `config` dictionary.

## See also

- <https://keras.io/api/models/model_saving_apis/serialization_utils#deserializekerasobject-function>

Other serialization utilities:  
[`get_custom_objects()`](https://keras3.posit.co/dev/reference/get_custom_objects.md)  
[`get_registered_name()`](https://keras3.posit.co/dev/reference/get_registered_name.md)  
[`get_registered_object()`](https://keras3.posit.co/dev/reference/get_registered_object.md)  
[`register_keras_serializable()`](https://keras3.posit.co/dev/reference/register_keras_serializable.md)  
[`serialize_keras_object()`](https://keras3.posit.co/dev/reference/serialize_keras_object.md)  
[`with_custom_object_scope()`](https://keras3.posit.co/dev/reference/with_custom_object_scope.md)  
