#' Define a custom `Layer` class.
#'
#' @description
#' A layer is a callable object that takes as input one or more tensors and
#' that outputs one or more tensors. It involves *computation*, defined
#' in the `call()` method, and a *state* (weight variables). State can be
#' created:
#'
#' * in `initialize()`, for instance via `self$add_weight()`;
#' * in the optional `build()` method, which is invoked by the first
#'   `call()` to the layer, and supplies the shape(s) of the input(s),
#'   which may not have been known at initialization time.
#'
#' Layers are recursively composable: If you assign a Layer instance as an
#' attribute of another Layer, the outer layer will start tracking the weights
#' created by the inner layer. Nested layers should be instantiated in the
#' `initialize()` method or `build()` method.
#'
#' Users will just instantiate a layer and then treat it as a callable.
#'
#' # Symbols in scope
#'
#' All R function custom methods (public and private) will have the
#' following symbols in scope:
#' * `self`: The custom class instance.
#' * `super`: The custom class superclass.
#' * `private`: An R environment specific to the class instance.
#'     Any objects assigned here are invisible to the Keras framework.
#' * `__class__` and `as.symbol(classname)`: the custom class type object.
#'
#' # Attributes
#'
#' * `name`: The name of the layer (string).
#' * `dtype`: Dtype of the layer's weights. Alias of `layer$variable_dtype`.
#' * `variable_dtype`: Dtype of the layer's weights.
#' * `compute_dtype`: The dtype of the layer's computations.
#'      Layers automatically cast inputs to this dtype, which causes
#'      the computations and output to also be in this dtype.
#'      When mixed precision is used with a
#'      `keras$mixed_precision$DTypePolicy`, this will be different
#'      than `variable_dtype`.
#' * `trainable_weights`: List of variables to be included in backprop.
#' * `non_trainable_weights`: List of variables that should not be
#'     included in backprop.
#' * `weights`: The concatenation of the lists `trainable_weights` and
#'     `non_trainable_weights` (in this order).
#' * `trainable`: Whether the layer should be trained (boolean), i.e.
#'     whether its potentially-trainable weights should be returned
#'     as part of `layer$trainable_weights`.
#' * `input_spec`: Optional (list of) `InputSpec` object(s) specifying the
#'     constraints on inputs that can be accepted by the layer.
#'
#' We recommend that custom `Layer`s implement the following methods:
#'
#' * `initialize()`: Defines custom layer attributes, and creates layer weights
#'     that do not depend on input shapes, using `add_weight()`,
#'     or other state.
#' * `build(input_shape)`: This method can be used to create weights that
#'     depend on the shape(s) of the input(s), using `add_weight()`, or other
#'     state. Calling `call()` will automatically build the layer
#'     (if it has not been built yet) by calling `build()`.
#' * `call(...)`: Method called after making
#'     sure `build()` has been called. `call()` performs the logic of applying
#'     the layer to the input arguments.
#'     Two reserved arguments you can optionally use in `call()` are:
#'
#'     1. `training` (boolean, whether the call is in inference mode or
#'         training mode).
#'     2. `mask` (boolean tensor encoding masked timesteps in the input,
#'         used e.g. in RNN layers).
#'
#'     A typical signature for this method is `call(inputs)`, and user
#'     could optionally add `training` and `mask` if the layer need them.
#' * `get_config()`: Returns a named list containing the configuration
#'     used to initialize this layer. If the list names differ from the arguments
#'     in `initialize()`, then override `from_config()` as well.
#'     This method is used when saving
#'     the layer or a model that contains this layer.
#'
#' # Examples
#' Here's a basic example: a layer with two variables, `w` and `b`,
#' that returns `y <- (w %*% x) + b`.
#' It shows how to implement `build()` and `call()`.
#' Variables set as attributes of a layer are tracked as weights
#' of the layers (in `layer$weights`).
#'
#' ```{r}
#' layer_simple_dense <- Layer(
#'   "SimpleDense",
#'   initialize = function(units = 32) {
#'     super$initialize()
#'     self$units <- units
#'   },
#'
#'   # Create the state of the layer (weights)
#'   build = function(input_shape) {
#'     self$kernel <- self$add_weight(
#'       shape = shape(tail(input_shape, 1), self$units),
#'       initializer = "glorot_uniform",
#'       trainable = TRUE,
#'       name = "kernel"
#'     )
#'     self$bias = self$add_weight(
#'       shape = shape(self$units),
#'       initializer = "zeros",
#'       trainable = TRUE,
#'       name = "bias"
#'     )
#'   },
#'
#'   # Defines the computation
#'   call = function(self, inputs) {
#'     op_matmul(inputs, self$kernel) + self$bias
#'   }
#' )
#'
#' # Instantiates the layer.
#' # Supply missing `object` arg to skip invoking `call()` and instead return
#' # the Layer instance
#' linear_layer <- layer_simple_dense(, 4)
#'
#' # This will call `build(input_shape)` and create the weights,
#' # and then invoke `call()`.
#' y <- linear_layer(op_ones(c(2, 2)))
#' stopifnot(length(linear_layer$weights) == 2)
#'
#' # These weights are trainable, so they're listed in `trainable_weights`:
#' stopifnot(length(linear_layer$trainable_weights) == 2)
#' ```
#'
#' Besides trainable weights, updated via backpropagation during training,
#' layers can also have non-trainable weights. These weights are meant to
#' be updated manually during `call()`. Here's a example layer that computes
#' the running sum of its inputs:
#'
#' ```{r}
#' layer_compute_sum <- Layer(
#'   classname = "ComputeSum",
#'
#'   initialize = function(input_dim) {
#'     super$initialize()
#'
#'     # Create a non-trainable weight.
#'     self$total <- self$add_weight(
#'       shape = shape(),
#'       initializer = "zeros",
#'       trainable = FALSE,
#'       name = "total"
#'     )
#'   },
#'
#'   call = function(inputs) {
#'     self$total$assign(self$total + op_sum(inputs))
#'     self$total
#'   }
#' )
#'
#' my_sum <- layer_compute_sum(, 2)
#' x <- op_ones(c(2, 2))
#' y <- my_sum(x)
#'
#' stopifnot(exprs = {
#'   all.equal(my_sum$weights,               list(my_sum$total))
#'   all.equal(my_sum$non_trainable_weights, list(my_sum$total))
#'   all.equal(my_sum$trainable_weights,     list())
#' })
#' ```
#'
#' @details
#'
#' # Methods available
#'
#' * ```r
#'   initialize(...,
#'              activity_regularizer = NULL,
#'              trainable = TRUE,
#'              dtype = NULL,
#'              autocast = TRUE,
#'              name = NULL)
#'   ```
#'   Initialize self. This method is typically called from a custom `initialize()` method.
#'   Example:
#'
#'   ```r
#'   layer_my_layer <- Layer("MyLayer",
#'     initialize = function(units, ..., dtype = NULL, name = NULL) {
#'       super$initialize(..., dtype = dtype, name = name)
#'       # .... finish initializing `self` instance
#'     }
#'   )
#'   ```
#'   Args:
#'   * trainable: Boolean, whether the layer's variables should be trainable.
#'   * name: String name of the layer.
#'   * dtype: The dtype of the layer's computations and weights. Can also be a
#'       `keras$DTypePolicy`,
#'       which allows the computation and
#'       weight dtype to differ. Defaults to `NULL`. `NULL` means to use
#'       `config_dtype_policy()`,
#'       which is a `"float32"` policy unless set to different value
#'       (via `config_set_dtype_policy()`).
#'
#' * ```r
#'   add_loss(loss)
#'   ```
#'   Can be called inside of the `call()` method to add a scalar loss.
#'
#'   Example:
#'
#'     ```r
#'     Layer("MyLayer",
#'       ...
#'       call = function(x) {
#'         self$add_loss(op_sum(x))
#'         x
#'       }
#'     ```
#'
#' * ```r
#'   add_metric(...)
#'   ```
#'
#' * ```r
#'   add_variable(...)
#'   ```
#'   Add a weight variable to the layer.
#'
#'   Alias of `add_weight()`.
#'
#' * ```r
#'   add_weight(shape = NULL,
#'              initializer = NULL,
#'              dtype = NULL,
#'              trainable = TRUE,
#'              autocast = TRUE,
#'              regularizer = NULL,
#'              constraint = NULL,
#'              aggregation = 'none',
#'              name = NULL)
#'   ```
#'   Add a weight variable to the layer.
#'
#'   Args:
#'   * `shape`: shape for the variable (as defined by [`shape()`])
#'       Must be fully-defined (no `NA`/`NULL`/`-1` entries).
#'       Defaults to `()` (scalar) if unspecified.
#'   * `initializer`: Initializer object to use to
#'       populate the initial variable value,
#'       or string name of a built-in initializer
#'       (e.g. `"random_normal"`). If unspecified,
#'       defaults to `"glorot_uniform"`
#'       for floating-point variables and to `"zeros"`
#'       for all other types (e.g. int, bool).
#'   * `dtype`: Dtype of the variable to create,
#'       e.g. `"float32"`. If unspecified,
#'       defaults to the layer's
#'       variable dtype (which itself defaults to
#'       `"float32"` if unspecified).
#'   * `trainable`: Boolean, whether the variable should
#'       be trainable via backprop or whether its
#'       updates are managed manually.
#'       Defaults to `TRUE`.
#'   * `autocast`: Boolean, whether to autocast layers variables when
#'       accessing them. Defaults to `TRUE`.
#'   * `regularizer`: Regularizer object to call to apply penalty on the
#'       weight. These penalties are summed into the loss function
#'       during optimization. Defaults to `NULL`.
#'   * `constraint`: Constraint object to call on the
#'       variable after any optimizer update,
#'       or string name of a built-in constraint.
#'       Defaults to `NULL`.
#'   * `aggregation`: Optional string, one of `NULL`, `"none"`, `"mean"`,
#'      `"sum"` or `"only_first_replica"`. Annotates the variable with
#'      the type of multi-replica aggregation to be used for this
#'      variable when writing custom data parallel training loops.
#'      Defaults to `"none"`.
#'   * `name`: String name of the variable. Useful for debugging purposes.
#'
#'   Returns:
#'
#'   A backend tensor, wrapped in a `KerasVariable` class.
#'   The `KerasVariable` class has
#'
#'     Methods:
#'     - `assign(value)`
#'     - `assign_add(value)`
#'     - `assign_sub(value)`
#'     - `numpy()` (calling `as.array(<variable>)` is preferred)
#'
#'     Properties/Attributes:
#'     - `value`
#'     - `dtype`
#'     - `ndim`
#'     - `shape` (calling `shape(<variable>)` is preferred)
#'     - `trainable`
#'
#' * ```r
#'   build(input_shape)
#'   ```
#'
#' * ```r
#'   build_from_config(config)
#'   ```
#'   Builds the layer's states with the supplied config (named list of args).
#'
#'   By default, this method calls the `do.call(build, config$input_shape)` method,
#'   which creates weights based on the layer's input shape in the supplied
#'   config. If your config contains other information needed to load the
#'   layer's state, you should override this method.
#'
#'   Args:
#'   * `config`: Named list containing the input shape associated with this layer.
#'
#' * ```r
#'   call(...)
#'   ```
#'   See description above
#'
#' * ```r
#'   compute_mask(inputs, previous_mask)
#'   ```
#'
#' * ```r
#'   compute_output_shape(...)
#'   ```
#'
#' * ```r
#'   compute_output_spec(...)
#'   ```
#'
#' * ```r
#'   count_params()
#'   ```
#'   Count the total number of scalars composing the weights.
#'
#'   Returns:
#'   An integer count.
#'
#'
#' * ```r
#'   get_build_config()
#'   ```
#'   Returns a named list with the layer's input shape.
#'
#'   This method returns a config (named list) that can be used by
#'   `build_from_config(config)` to create all states (e.g. Variables and
#'   Lookup tables) needed by the layer.
#'
#'   By default, the config only contains the input shape that the layer
#'   was built with. If you're writing a custom layer that creates state in
#'   an unusual way, you should override this method to make sure this state
#'   is already created when Keras attempts to load its value upon model
#'   loading.
#'
#'   Returns:
#'   A named list containing the input shape associated with the layer.
#'
#' * ```r
#'   get_config()
#'   ```
#'   Returns the config of the object.
#'
#'   An object config is a named list (serializable)
#'   containing the information needed to re-instantiate it.
#'   The config is expected to be serializable to JSON, and is expected
#'   to consist of a (potentially complex, nested) structure of names lists
#'   consisting of simple objects like strings, ints.
#'
#' * ```r
#'   get_weights()
#'   ```
#'   Return the values of `layer$weights` as a list of R or NumPy arrays.
#'
#' * ```r
#'   quantize(mode, type_check = TRUE)
#'   ```
#'   Currently, only the `Dense`, `EinsumDense` and `Embedding` layers support in-place
#'   quantization via this `quantize()` method.
#'
#'   Example:
#'   ```r
#'   model$quantize("int8") # quantize model in-place
#'   model |> predict(data) # faster inference
#'   ```
#'
#' * ```r
#'   quantized_build(input_shape, mode)
#'   ```
#'
#' * ```r
#'   quantized_call(...)
#'   ```
#'
#' * ```r
#'   load_own_variables(store)
#'   ```
#'   Loads the state of the layer.
#'
#'   You can override this method to take full control of how the state of
#'   the layer is loaded upon calling `load_model()`.
#'
#'   Args:
#'   * `store`: Named list from which the state of the model will be loaded.
#'
#' * ```r
#'   save_own_variables(store)
#'   ```
#'   Saves the state of the layer.
#'
#'   You can override this method to take full control of how the state of
#'   the layer is saved upon calling `save_model()`.
#'
#'   Args:
#'   * `store`: Named list where the state of the model will be saved.
#'
#' * ```r
#'   set_weights(weights)
#'   ```
#'   Sets the values of `weights` from a list of R or NumPy arrays.
#'
#' * ```r
#'   stateless_call(trainable_variables, non_trainable_variables,
#'                  ..., return_losses = FALSE)
#'   ```
#'   Call the layer without any side effects.
#'
#'   Args:
#'   * `trainable_variables`: List of trainable variables of the model.
#'   * `non_trainable_variables`: List of non-trainable variables of the
#'        model.
#'   * `...`: Positional and named arguments to be passed to `call()`.
#'   * `return_losses`: If `TRUE`, `stateless_call()` will return the list of
#'        losses created during `call()` as part of its return values.
#'
#'   Returns:
#'   An unnamed list. By default, returns `list(outputs, non_trainable_variables)`.
#'   If `return_losses = TRUE`, then returns
#'   `list(outputs, non_trainable_variables, losses)`.
#'
#'   Note: `non_trainable_variables` include not only non-trainable weights
#'   such as `BatchNormalization` statistics, but also RNG seed state
#'   (if there are any random operations part of the layer, such as dropout),
#'   and `Metric` state (if there are any metrics attached to the layer).
#'   These are all elements of state of the layer.
#'
#'   Example:
#'
#'   ```r
#'   model <- ...
#'   data <- ...
#'   trainable_variables <- model$trainable_variables
#'   non_trainable_variables <- model$non_trainable_variables
#'   # Call the model with zero side effects
#'   c(outputs, non_trainable_variables) %<-% model$stateless_call(
#'       trainable_variables,
#'       non_trainable_variables,
#'       data
#'   )
#'   # Attach the updated state to the model
#'   # (until you do this, the model is still in its pre-call state).
#'   purrr::walk2(
#'     model$non_trainable_variables, non_trainable_variables,
#'     \(variable, value) variable$assign(value))
#'   ```
#'
#' * ```r
#'   symbolic_call(...)
#'   ```
#'
#' * ```r
#'   from_config(config)
#'   ```
#'
#'   Creates a layer from its config.
#'
#'   This is a class method, meaning, the R function will not have a `self`
#'   symbol (a class instance) in scope. Use `__class__` or the classname symbol
#'   provided when the `Layer()` was constructed) to resolve the class definition.
#'   The default implementation is:
#'   ```r
#'   from_config = function(config) {
#'     do.call(`__class__`, config)
#'   }
#'   ```
#'
#'   This method is the reverse of `get_config()`,
#'   capable of instantiating the same layer from the config
#'   named list. It does not handle layer connectivity
#'   (handled by Network), nor weights (handled by `set_weights()`).
#'
#'   Args:
#'   * `config`: A named list, typically the
#'      output of `get_config()`.
#'
#'   Returns:
#'   A layer instance.
#'
#'
#' # Readonly properties:
#'
#' * `compute_dtype`
#'     The dtype of the computations performed by the layer.
#'
#' * `dtype`
#'     Alias of `layer$variable_dtype`.
#'
#' * `input_dtype`
#'     The dtype layer inputs should be converted to.
#'
#' * `losses`
#'     List of scalar losses from `add_loss()`, regularizers and sublayers.
#'
#' * `metrics`
#'     List of all metrics.
#'
#' * `metrics_variables`
#'     List of all metric variables.
#'
#' * `non_trainable_variables`
#'     List of all non-trainable layer state.
#'
#'     This extends `layer$non_trainable_weights` to include all state used by
#'     the layer including state for metrics and `SeedGenerator`s.
#'
#' * `non_trainable_weights`
#'     List of all non-trainable weight variables of the layer.
#'
#'     These are the weights that should not be updated by the optimizer during
#'     training. Unlike, `layer$non_trainable_variables` this excludes metric
#'     state and random seeds.
#'
#' * `trainable_variables`
#'     List of all trainable layer state.
#'
#'     This is equivalent to `layer$trainable_weights`.
#'
#' * `trainable_weights`
#'     List of all trainable weight variables of the layer.
#'
#'     These are the weights that get updated by the optimizer during training.
#'
#' * `path`
#'     The path of the layer.
#'
#'     If the layer has not been built yet, it will be `NULL`.
#'
#' * `quantization_mode`
#'     The quantization mode of this layer, `NULL` if not quantized.
#'
#' * `variable_dtype`
#'     The dtype of the state (weights) of the layer.
#'
#' * `variables`
#'     List of all layer state, including random seeds.
#'
#'     This extends `layer$weights` to include all state used by the layer
#'     including `SeedGenerator`s.
#'
#'     Note that metrics variables are not included here, use
#'     `metrics_variables` to visit all the metric variables.
#'
#' * `weights`
#'     List of all weight variables of the layer.
#'
#'     Unlike, `layer$variables` this excludes metric state and random seeds.
#'
#' * `input`
#'     Retrieves the input tensor(s) of a symbolic operation.
#'
#'     Only returns the tensor(s) corresponding to the *first time*
#'     the operation was called.
#'
#'     Returns:
#'     Input tensor or list of input tensors.
#'
#' * `output`
#'    Retrieves the output tensor(s) of a layer.
#'
#'    Only returns the tensor(s) corresponding to the *first time*
#'    the operation was called.
#'
#'    Returns:
#'    Output tensor or list of output tensors.
#'
#' # Data descriptors (Attributes):
#'
#' * `dtype_policy`
#'
#' * `input_spec`
#'
#' * `supports_masking`
#'    Whether this layer supports computing a mask using `compute_mask`.
#'
#' * `trainable`
#'    Settable boolean, whether this layer should be trainable or not.
#'
#' @param classname String, the name of the custom class. (Conventionally, CamelCase).
#' @param initialize,call,build,get_config Recommended methods to implement. See
#'   description and details sections.
#' @param ...,public Additional methods or public members of the custom class.
#' @param private Named list of R objects (typically, functions) to include in
#'   instance private environments. `private` methods will have all the same
#'   symbols in scope as public methods (See section "Symbols in Scope"). Each
#'   instance will have it's own `private` environment. Any objects
#'   in `private` will be invisible from the Keras framework and the Python
#'   runtime.
#' @param parent_env The R environment that all class methods will have as a grandparent.
#' @param inherit What the custom class will subclass. By default, the base keras class.
#'
#' @returns A composing layer constructor, with similar behavior to other layer
#' functions like `layer_dense()`. The first argument of the returned function
#' will be `object`, enabling `initialize()`ing and `call()` the layer in one
#' step while composing the layer with the pipe, like
#'
#' ```r
#' layer_foo <- Layer("Foo", ....)
#' output <- inputs |> layer_foo()
#' ```
#' To only `initialize()` a layer instance and not `call()` it, pass a missing
#' or `NULL` value to `object`, or pass all arguments to `initialize()` by name.
#'
#' ```r
#' layer <- layer_dense(units = 2, activation = "relu")
#' layer <- layer_dense(NULL, 2, activation = "relu")
#' layer <- layer_dense(, 2, activation = "relu")
#'
#' # then you can call() the layer in a separate step
#' outputs <- inputs |> layer()
#' ```
#'
#' @tether keras.layers.Layer
#' @export
#' @family layers
#' @importFrom utils modifyList
#' @seealso
#' + <https://keras.io/api/layers/base_layer#layer-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>
Layer <-
function(classname,
         initialize = NULL,
         call = NULL,
         build = NULL,
         get_config = NULL,
         ...,
         public = list(),
         private = list(),
         inherit = NULL,
         parent_env = parent.frame()) {

  members <- drop_nulls(named_list(initialize, call, build, get_config))
  members <- modifyList(members, list2(...), keep.null = TRUE)
  members <- modifyList(members, public, keep.null = TRUE)

  members <- modify_intersection(members, list(
    from_config = function(x) decorate_method(x, "classmethod")
  ))

  inherit <- substitute(inherit) %||%
    quote(base::asNamespace("keras3")$keras$Layer)

  wrapper <- new_wrapped_py_class(
    classname = classname,
    members = members,
    inherit = inherit,
    parent_env = parent_env,
    private = private
  )

  # convert wrapper into a composing layer rapper
  prepend(formals(wrapper)) <- alist(object = )
  body(wrapper) <-  bquote({
    args <- capture_args(ignore = "object",
                          enforce_all_dots_named = FALSE)
    create_layer(.(as.symbol(classname)), object, args)
  })

  wrapper
}



# ' @param .composing Bare Keras Layers (`layer_*` functions) conventionally
# have `object` as the first argument, which allows users to instantiate
# (`initialize`) and `call` one motion.


