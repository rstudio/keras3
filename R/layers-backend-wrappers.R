

#' Torch module wrapper layer.
#'
#' @description
#' `layer_torch_module_wrapper` is a wrapper class that can turn any
#' `torch.nn.Module` into a Keras layer, in particular by making its
#' parameters trackable by Keras.
#'
#' `layer_torch_module_wrapper()` is only compatible with the PyTorch backend and
#' cannot be used with the TensorFlow or JAX backends.
#'
#' # Example
#' Here's an example of how the [`layer_torch_module_wrapper()`] can be used with vanilla
#' PyTorch modules.
#'
#' ```r
#' # reticulate::py_install(
#' #   packages = c("torch", "torchvision", "torchaudio"),
#' #   envname = "r-keras",
#' #   pip_options = c("--index-url https://download.pytorch.org/whl/cpu")
#' # )
#' library(keras3)
#' use_backend("torch")
#' torch <- reticulate::import("torch")
#' nn <- reticulate::import("torch.nn")
#' nnf <- reticulate::import("torch.nn.functional")
#'
#' Classifier(keras$Model) \%py_class\% {
#'   initialize <- function(...) {
#'     super$initialize(...)
#'
#'     self$conv1 <- layer_torch_module_wrapper(module = nn$Conv2d(
#'       in_channels = 1L,
#'       out_channels = 32L,
#'       kernel_size = tuple(3L, 3L)
#'     ))
#'     self$conv2 <- layer_torch_module_wrapper(module = nn$Conv2d(
#'       in_channels = 32L,
#'       out_channels = 64L,
#'       kernel_size = tuple(3L, 3L)
#'     ))
#'     self$pool <- nn$MaxPool2d(kernel_size = tuple(2L, 2L))
#'     self$flatten <- nn$Flatten()
#'     self$dropout <- nn$Dropout(p = 0.5)
#'     self$fc <-
#'       layer_torch_module_wrapper(module = nn$Linear(1600L, 10L))
#'   }
#'
#'   call <- function(inputs) {
#'     x <- nnf$relu(self$conv1(inputs))
#'     x <- self$pool(x)
#'     x <- nnf$relu(self$conv2(x))
#'     x <- self$pool(x)
#'     x <- self$flatten(x)
#'     x <- self$dropout(x)
#'     x <- self$fc(x)
#'     nnf$softmax(x, dim = 1L)
#'   }
#' }
#' model <- Classifier()
#' model$build(shape(1, 28, 28))
#' cat("Output shape:", format(shape(model(torch$ones(1L, 1L, 28L, 28L)))))
#'
#' model |> compile(loss = "sparse_categorical_crossentropy",
#'                  optimizer = "adam",
#'                  metrics = "accuracy")
#' ```
#' ```r
#' model |> fit(train_loader, epochs = 5)
#' ```
#'
#' @param module
#' `torch.nn.Module` instance. If it's a `LazyModule`
#' instance, then its parameters must be initialized before
#' passing the instance to `layer_torch_module_wrapper` (e.g. by calling
#' it once).
#'
#' @param name
#' The name of the layer (string).
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit layer_dense return
#' @export
#' @family wrapping layers
#' @family layers
#' @tether keras.layers.TorchModuleWrapper
layer_torch_module_wrapper <-
function (object, module, name = NULL, ...)
{
    args <- capture_args(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$TorchModuleWrapper, object, args)
}


#' Keras Layer that wraps a [Flax](https://flax.readthedocs.io) module.
#'
#' @description
#' This layer enables the use of Flax components in the form of
#' [`flax.linen.Module`](
#'     https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html)
#' instances within Keras when using JAX as the backend for Keras.
#'
#' The module method to use for the forward pass can be specified via the
#' `method` argument and is `__call__` by default. This method must take the
#' following arguments with these exact names:
#'
#' - `self` if the method is bound to the module, which is the case for the
#'     default of `__call__`, and `module` otherwise to pass the module.
#' - `inputs`: the inputs to the model, a JAX array or a `PyTree` of arrays.
#' - `training` *(optional)*: an argument specifying if we're in training mode
#'     or inference mode, `TRUE` is passed in training mode.
#'
#' `FlaxLayer` handles the non-trainable state of your model and required RNGs
#' automatically. Note that the `mutable` parameter of
#' [`flax.linen.Module.apply()`](
#'     https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.apply)
#' is set to `DenyList(["params"])`, therefore making the assumption that all
#' the variables outside of the "params" collection are non-trainable weights.
#'
#' This example shows how to create a `FlaxLayer` from a Flax `Module` with
#' the default `__call__` method and no training argument:
#'
#' ```{r}
#' # keras3::use_backend("jax")
#' # py_install("flax", "r-keras")
#'
#' if(config_backend() == "jax" &&
#'    reticulate::py_module_available("flax")) {
#'
#' flax <- import("flax")
#'
#' MyFlaxModule(flax$linen$Module) %py_class% {
#'   `__call__` <- flax$linen$compact(\(self, inputs) {
#'     inputs |>
#'       (flax$linen$Conv(features = 32L, kernel_size = tuple(3L, 3L)))() |>
#'       flax$linen$relu() |>
#'       flax$linen$avg_pool(window_shape = tuple(2L, 2L),
#'                           strides = tuple(2L, 2L)) |>
#'       # flatten all except batch_size axis
#'       (\(x) x$reshape(tuple(x$shape[[1]], -1L)))() |>
#'       (flax$linen$Dense(features = 200L))() |>
#'       flax$linen$relu() |>
#'       (flax$linen$Dense(features = 10L))() |>
#'       flax$linen$softmax()
#'   })
#' }
#'
#' # typical usage:
#' input <- keras_input(c(28, 28, 3))
#' output <- input |>
#'   layer_flax_module_wrapper(MyFlaxModule())
#'
#' model <- keras_model(input, output)
#'
#' # to instantiate the layer before composing:
#' flax_module <- MyFlaxModule()
#' keras_layer <- layer_flax_module_wrapper(module = flax_module)
#'
#' input <- keras_input(c(28, 28, 3))
#' output <- input |>
#'   keras_layer()
#'
#' model <- keras_model(input, output)
#'
#' }
#' ```
#'
#' This example shows how to wrap the module method to conform to the required
#' signature. This allows having multiple input arguments and a training
#' argument that has a different name and values. This additionally shows how
#' to use a function that is not bound to the module.
#'
#' ```r
#' flax <- import("flax")
#'
#' MyFlaxModule(flax$linen$Module) \%py_class\% {
#'   forward <-
#'     flax$linen$compact(\(self, inputs1, input2, deterministic) {
#'       # do work ....
#'       outputs # return
#'     })
#' }
#'
#' my_flax_module_wrapper <- function(module, inputs, training) {
#'   c(input1, input2) \%<-\% inputs
#'   module$forward(input1, input2,!training)
#' }
#'
#' flax_module <- MyFlaxModule()
#' keras_layer <- layer_flax_module_wrapper(module = flax_module,
#'                                          method = my_flax_module_wrapper)
#' ```
#'
#' @param module
#' An instance of `flax.linen.Module` or subclass.
#'
#' @param method
#' The method to call the model. This is generally a method in the
#' `Module`. If not provided, the `__call__` method is used. `method`
#' can also be a function not defined in the `Module`, in which case it
#' must take the `Module` as the first argument. It is used for both
#' `Module.init` and `Module.apply`. Details are documented in the
#' `method` argument of [`flax.linen.Module.apply()`](
#'   https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.apply).
#'
#' @param variables
#' A `dict` (named R list) containing all the variables of the module in the
#' same format as what is returned by [`flax.linen.Module.init()`](
#'   https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.init).
#' It should contain a `"params"` key and, if applicable, other keys for
#' collections of variables for non-trainable state. This allows
#' passing trained parameters and learned non-trainable state or
#' controlling the initialization. If `NULL` is passed, the module's
#' `init` function is called at build time to initialize the variables
#' of the model.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit layer_dense return
#' @export
#' @family wrapping layers
#' @family layers
#' @tether keras.layers.FlaxLayer
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/FlaxLayer>
layer_flax_module_wrapper <-
function (object, module, method = NULL, variables = NULL, ...)
{
    args <- capture_args(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$FlaxLayer, object, args)
}


#' Keras Layer that wraps a JAX model.
#'
#' @description
#' This layer enables the use of JAX components within Keras when using JAX as
#' the backend for Keras.
#'
#' # Model function
#'
#' This layer accepts JAX models in the form of a function, `call_fn()`, which
#' must take the following arguments with these exact names:
#'
#' - `params`: trainable parameters of the model.
#' - `state` (*optional*): non-trainable state of the model. Can be omitted if
#'     the model has no non-trainable state.
#' - `rng` (*optional*): a `jax.random.PRNGKey` instance. Can be omitted if the
#'     model does not need RNGs, neither during training nor during inference.
#' - `inputs`: inputs to the model, a JAX array or a `PyTree` of arrays.
#' - `training` (*optional*): an argument specifying if we're in training mode
#'     or inference mode, `TRUE` is passed in training mode. Can be omitted if
#'     the model behaves the same in training mode and inference mode.
#'
#' The `inputs` argument is mandatory. Inputs to the model must be provided via
#' a single argument. If the JAX model takes multiple inputs as separate
#' arguments, they must be combined into a single structure, for instance in a
#' `tuple()` or a `dict()`.
#'
#' ## Model weights initialization
#'
#' The initialization of the `params` and `state` of the model can be handled
#' by this layer, in which case the `init_fn()` argument must be provided. This
#' allows the model to be initialized dynamically with the right shape.
#' Alternatively, and if the shape is known, the `params` argument and
#' optionally the `state` argument can be used to create an already initialized
#' model.
#'
#' The `init_fn()` function, if provided, must take the following arguments with
#' these exact names:
#'
#' - `rng`: a `jax.random.PRNGKey` instance.
#' - `inputs`: a JAX array or a `PyTree` of arrays with placeholder values to
#'     provide the shape of the inputs.
#' - `training` (*optional*): an argument specifying if we're in training mode
#'     or inference mode. `True` is always passed to `init_fn`. Can be omitted
#'     regardless of whether `call_fn` has a `training` argument.
#'
#' ## Models with non-trainable state
#'
#' For JAX models that have non-trainable state:
#'
#' - `call_fn()` must have a `state` argument
#' - `call_fn()` must return a `tuple()` containing the outputs of the model and
#'     the new non-trainable state of the model
#' - `init_fn()` must return a `tuple()` containing the initial trainable params of
#'     the model and the initial non-trainable state of the model.
#'
#' This code shows a possible combination of `call_fn()` and `init_fn()` signatures
#' for a model with non-trainable state. In this example, the model has a
#' `training` argument and an `rng` argument in `call_fn()`.
#'
#' ```r
#' stateful_call <- function(params, state, rng, inputs, training) {
#'   outputs <- ....
#'   new_state <- ....
#'   tuple(outputs, new_state)
#' }
#'
#' stateful_init <- function(rng, inputs) {
#'   initial_params <- ....
#'   initial_state <- ....
#'   tuple(initial_params, initial_state)
#' }
#' ```
#' ## Models without non-trainable state
#'
#' For JAX models with no non-trainable state:
#'
#' - `call_fn()` must not have a `state` argument
#' - `call_fn()` must return only the outputs of the model
#' - `init_fn()` must return only the initial trainable params of the model.
#'
#' This code shows a possible combination of `call_fn()` and `init_fn()` signatures
#' for a model without non-trainable state. In this example, the model does not
#' have a `training` argument and does not have an `rng` argument in `call_fn()`.
#'
#' ```r
#' stateful_call <- function(pparams, inputs) {
#'   outputs <- ....
#'   outputs
#' }
#'
#' stateful_init <- function(rng, inputs) {
#'   initial_params <- ....
#'   initial_params
#' }
#' ```
#'
#' ## Conforming to the required signature
#'
#' If a model has a different signature than the one required by `JaxLayer`,
#' one can easily write a wrapper method to adapt the arguments. This example
#' shows a model that has multiple inputs as separate arguments, expects
#' multiple RNGs in a `dict`, and has a `deterministic` argument with the
#' opposite meaning of `training`. To conform, the inputs are combined in a
#' single structure using a `tuple`, the RNG is split and used the populate the
#' expected `dict`, and the Boolean flag is negated:
#'
#' ```r
#' jax <- import("jax")
#' my_model_fn <- function(params, rngs, input1, input2, deterministic) {
#'   ....
#'   if (!deterministic) {
#'     dropout_rng <- rngs$dropout
#'     keep <- jax$random$bernoulli(dropout_rng, dropout_rate, x$shape)
#'     x <- jax$numpy$where(keep, x / dropout_rate, 0)
#'     ....
#'   }
#'   ....
#'   return(outputs)
#' }
#'
#' my_model_wrapper_fn <- function(params, rng, inputs, training) {
#'   c(input1, input2) %<-% inputs
#'   c(rng1, rng2) %<-% jax$random$split(rng)
#'   rngs <-  list(dropout = rng1, preprocessing = rng2)
#'   deterministic <-  !training
#'   my_model_fn(params, rngs, input1, input2, deterministic)
#' }
#'
#' keras_layer <- layer_jax_model_wrapper(call_fn = my_model_wrapper_fn,
#'                                        params = initial_params)
#' ```
#'
#' ## Usage with Haiku modules
#'
#' `JaxLayer` enables the use of [Haiku](https://dm-haiku.readthedocs.io)
#' components in the form of
#' [`haiku.Module`](https://dm-haiku.readthedocs.io/en/latest/api.html#module).
#' This is achieved by transforming the module per the Haiku pattern and then
#' passing `module.apply` in the `call_fn` parameter and `module.init` in the
#' `init_fn` parameter if needed.
#'
#' If the model has non-trainable state, it should be transformed with
#' [`haiku.transform_with_state`](
#'   https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.transform_with_state).
#' If the model has no non-trainable state, it should be transformed with
#' [`haiku.transform`](
#'   https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.transform).
#' Additionally, and optionally, if the module does not use RNGs in "apply", it
#' can be transformed with
#' [`haiku.without_apply_rng`](
#'   https://dm-haiku.readthedocs.io/en/latest/api.html#without-apply-rng).
#'
#' The following example shows how to create a `JaxLayer` from a Haiku module
#' that uses random number generators via `hk.next_rng_key()` and takes a
#' training positional argument:
#'
#' ```r
#' # reticulate::py_install("haiku", "r-keras")
#' hk <- import("haiku")
#' MyHaikuModule(hk$Module) \%py_class\% {
#'
#'   `__call__` <- \(self, x, training) {
#'     x <- hk$Conv2D(32L, tuple(3L, 3L))(x)
#'     x <- jax$nn$relu(x)
#'     x <- hk$AvgPool(tuple(1L, 2L, 2L, 1L),
#'                     tuple(1L, 2L, 2L, 1L), "VALID")(x)
#'     x <- hk$Flatten()(x)
#'     x <- hk$Linear(200L)(x)
#'     if (training)
#'       x <- hk$dropout(rng = hk$next_rng_key(), rate = 0.3, x = x)
#'     x <- jax$nn$relu(x)
#'     x <- hk$Linear(10L)(x)
#'     x <- jax$nn$softmax(x)
#'     x
#'   }
#'
#' }
#'
#' my_haiku_module_fn <- function(inputs, training) {
#'   module <- MyHaikuModule()
#'   module(inputs, training)
#' }
#'
#' transformed_module <- hk$transform(my_haiku_module_fn)
#'
#' keras_layer <-
#'   layer_jax_model_wrapper(call_fn = transformed_module$apply,
#'                           init_fn = transformed_module$init)
#' ```
#'
#' @param call_fn
#' The function to call the model. See description above for the
#' list of arguments it takes and the outputs it returns.
#'
#' @param init_fn
#' the function to call to initialize the model. See description
#' above for the list of arguments it takes and the outputs it returns.
#' If `NULL`, then `params` and/or `state` must be provided.
#'
#' @param params
#' A `PyTree` containing all the model trainable parameters. This
#' allows passing trained parameters or controlling the initialization.
#' If both `params` and `state` are `NULL`, `init_fn()` is called at
#' build time to initialize the trainable parameters of the model.
#'
#' @param state
#' A `PyTree` containing all the model non-trainable state. This
#' allows passing learned state or controlling the initialization. If
#' both `params` and `state` are `NULL`, and `call_fn()` takes a `state`
#' argument, then `init_fn()` is called at build time to initialize the
#' non-trainable state of the model.
#'
#' @param seed
#' Seed for random number generator. Optional.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit layer_dense return
#' @export
#' @family wrapping layers
#' @family layers
#' @tether keras.layers.JaxLayer
layer_jax_model_wrapper <-
function (object, call_fn, init_fn = NULL, params = NULL, state = NULL,
    seed = NULL, ...)
{
    args <- capture_args(list(seed = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$JaxLayer, object, args)
}
