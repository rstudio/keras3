# Clone a Functional or Sequential `Model` instance.

Model cloning is similar to calling a model on new inputs, except that
it creates new layers (and thus new weights) instead of sharing the
weights of the existing layers.

Note that `clone_model()` will not preserve the uniqueness of shared
objects within the model (e.g. a single variable attached to two
distinct layers will be restored as two separate variables).

## Usage

``` r
clone_model(
  model,
  input_tensors = NULL,
  clone_function = NULL,
  call_function = NULL,
  recursive = FALSE,
  ...
)
```

## Arguments

- model:

  Instance of `Model` (could be a Functional model or a Sequential
  model).

- input_tensors:

  Optional list of input tensors to build the model upon. If not
  provided, new
  [`keras_input()`](https://keras3.posit.co/dev/reference/keras_input.md)
  objects will be created.

- clone_function:

  Callable with signature `function(layer)` to be used to clone each
  layer in the target model (except `Input` instances). It takes as
  argument the layer instance to be cloned, and returns the
  corresponding layer instance to be used in the model copy. If
  unspecified, this callable defaults to the following
  serialization/deserialization function:
  `` function(layer) layer$`__class__`$from_config(layer$get_config()) ``.
  By passing a custom callable, you can customize your copy of the
  model, e.g. by wrapping certain layers of interest (you might want to
  replace all `LSTM` instances with equivalent
  `Bidirectional(LSTM(...))` instances, for example). Defaults to
  `NULL`.

- call_function:

  Callable with signature `function(layer, ...)` to be used to call each
  cloned layer and a set of inputs. It takes the layer instance, and the
  call arguments, and returns the call outputs. If unspecified, this
  callable defaults to the regular
  [`call()`](https://rdrr.io/r/base/call.html) method:
  `function(layer, ...) do.call(layer, list(...))`. By passing a custom
  callable, you can insert new layers before or after a given layer.

- recursive:

  Note, This argument can only be used with Functional models. Boolean.
  Whether to recursively clone any Sequential or Functional models
  encountered in the original Sequential/Functional model. If `FALSE`,
  then inner models are cloned by calling `clone_function()`. If `TRUE`,
  then inner models are cloned by calling `clone_model()` with the same
  `clone_function`, `call_function`, and `recursive` arguments. Note
  that in this case, `call_function` will not be propagated to any
  Sequential model (since it is not applicable to Sequential models).

- ...:

  For forward/backward compatability.

## Value

An instance of `Model` reproducing the behavior of the original model,
on top of new inputs tensors, using newly instantiated weights. The
cloned model may behave differently from the original model if a custom
`clone_function` or `call_function` modifies a layer or layer call.

## Examples

    # Create a test Sequential model.
    model <- keras_model_sequential(input_shape = c(728)) |>
      layer_dense(32, activation = 'relu') |>
      layer_dense(1, activation = 'sigmoid')

    # Create a copy of the test model (with freshly initialized weights).
    new_model <- clone_model(model)

Using a `clone_function` to make a model deterministic by setting the
random seed everywhere:

    clone_function <- function(layer) {
      config <- layer$get_config()
      if ("seed" %in% names(config))
        config$seed <- 1337L
      layer$`__class__`$from_config(config)
    }

    new_model <- clone_model(model, clone_function = clone_function)

Using a `call_function` to add a `Dropout` layer after each `Dense`
layer (without recreating new layers):

    call_function <- function(layer, ...) {
      out <- layer(...)
      if (inherits(layer, keras$layers$Dense))
        out <- out |> layer_dropout(0.5)
      out
    }

    inputs <- keras_input(c(728))
    outputs <- inputs |>
      layer_dense(32, activation = 'relu') |>
      layer_dense(1, activation = 'sigmoid')
    model <- keras_model(inputs, outputs)

    new_model <- clone_model(
      model,
      clone_function = function(x) x, # Reuse the same layers.
      call_function = call_function,
    )
    new_model

    ## Model: "functional_4"
    ## +-----------------------------------+--------------------------+---------------+
    ## | Layer (type)                      | Output Shape             |       Param # |
    ## +===================================+==========================+===============+
    ## | keras_tensor_8 (InputLayer)       | (None, 728)              |             0 |
    ## +-----------------------------------+--------------------------+---------------+
    ## | dense_2 (Dense)                   | (None, 32)               |        23,328 |
    ## +-----------------------------------+--------------------------+---------------+
    ## | dropout (Dropout)                 | (None, 32)               |             0 |
    ## +-----------------------------------+--------------------------+---------------+
    ## | dense_3 (Dense)                   | (None, 1)                |            33 |
    ## +-----------------------------------+--------------------------+---------------+
    ## | dropout_1 (Dropout)               | (None, 1)                |             0 |
    ## +-----------------------------------+--------------------------+---------------+
    ##  Total params: 23,361 (91.25 KB)
    ##  Trainable params: 23,361 (91.25 KB)
    ##  Non-trainable params: 0 (0.00 B)

Note that subclassed models cannot be cloned by default, since their
internal layer structure is not known. To achieve equivalent
functionality as `clone_model` in the case of a subclassed model, simply
make sure that the model class implements
[`get_config()`](https://keras3.posit.co/dev/reference/get_config.md)
(and optionally
[`from_config()`](https://keras3.posit.co/dev/reference/get_config.md)),
and call:

    new_model <- model$`__class__`$from_config(model$get_config())

In the case of a subclassed model, you cannot using a custom
`clone_function`.
