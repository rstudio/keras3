

#' Torch module wrapper layer.
#'
#' @description
#' `layer_torch_module_wrapper` is a wrapper class that can turn any
#' `torch.nn.Module` into a Keras layer, in particular by making its
#' parameters trackable by Keras.
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
#' @family utils
#' @family layers
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/TorchModuleWrapper>
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
#'   layer_flax(MyFlaxModule())
#'
#' model <- keras_model(input, output)
#'
#' # to instantiate the layer before composing:
#' flax_module <- MyFlaxModule()
#' keras_layer <- layer_flax(module = flax_module)
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
#' MyFlaxModule(flax$linen$Module) %py_class% {
#'   forward <-
#'     flax$linen$compact(\(self, inputs1, input2, deterministic) {
#'       # do work ....
#'       outputs # return
#'     })
#' }
#'
#' my_flax_module_wrapper <- function(module, inputs, training) {
#'   c(input1, input2) %<-% inputs
#'   module$forward(input1, input2,!training)
#' }
#'
#' flax_module <- MyFlaxModule()
#' keras_layer <- FlaxLayer(module = flax_module,
#'                          method = my_flax_module_wrapper)
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
#' @export
#' @tether keras.layers.FlaxLayer
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/FlaxLayer>
layer_flax <-
function (object, module, method = NULL, variables = NULL, ...)
{
    args <- capture_args(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$FlaxLayer, object, args)
}

