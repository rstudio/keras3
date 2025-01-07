


#' Publicly accessible method for determining the current backend.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' config_backend()
#' ```
#'
#' @returns
#' String, the name of the backend Keras is currently using. One of
#' `"tensorflow"`, `"torch"`, or `"jax"`.
#'
#' @export
#' @family config backend
#' @family backend
#' @family config
#' @seealso
#' [use_backend()]
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/backend>
#' @tether keras.config.backend
config_backend <-
function ()
{
    keras$config$backend()
}

#' Reload the backend (and the Keras package).
#'
#' @description
#'
#' # Examples
#' ```python
#' config_set_backend("jax")
#' ```
#'
#' # WARNING
#' Using this function is dangerous and should be done
#' carefully. Changing the backend will **NOT** convert
#' the type of any already-instantiated objects.
#' Thus, any layers / tensors / etc. already created will no
#' longer be usable without errors. It is strongly recommended **not**
#' to keep around **any** Keras-originated objects instances created
#' before calling `config_set_backend()`.
#'
#' This includes any function or class instance that uses any Keras
#' functionality. All such code needs to be re-executed after calling
#' `config_set_backend()`.
#'
#' @param backend String
#'
#' @returns Nothing, this function is called for its side effect.
#'
#' @family config
#' @export
#' @tether keras.config.set_backend
config_set_backend <-
function (backend)
{
  if(!is_keras_loaded())
    return(use_backend(backend))
  keras$config$set_backend(backend)
  invisible(backend)
}


#' Return the value of the fuzz factor used in numeric expressions.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' config_epsilon()
#' ```
#'
#' @returns
#' A float.
#'
#' @export
#' @family config backend
#' @family backend
#' @family config
#' @seealso
#' + <https://keras.io/api/utils/config_utils#epsilon-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/epsilon>
#' @tether keras.config.epsilon
config_epsilon <-
function ()
{
    keras$config$epsilon()
}


#' Return the default float type, as a string.
#'
#' @description
#' E.g. `'bfloat16'` `'float16'`, `'float32'`, `'float64'`.
#'
#' # Examples
#' ```{r}
#' keras3::config_floatx()
#' ```
#'
#' @returns
#' String, the current default float type.
#'
#' @export
#' @family config backend
#' @family backend
#' @family config
#' @seealso
#' + <https://keras.io/api/utils/config_utils#floatx-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/floatx>
#' @tether keras.config.floatx
config_floatx <-
function ()
{
   keras$config$floatx()
}


#??
function(x) {
  # config_floatx?
  if(missing(x))
    keras$config$floatx()
  else
    keras$config$set_floatx(x)
}




#' Return the default image data format convention.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' config_image_data_format()
#' ```
#'
#' @returns
#' A string, either `'channels_first'` or `'channels_last'`.
#'
#' @export
#' @family config backend
#' @family backend
#' @family config
#' @seealso
#' + <https://keras.io/api/utils/config_utils#imagedataformat-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/image_data_format>
#' @tether keras.config.image_data_format
config_image_data_format <-
function ()
{
    args <- capture_args()
    do.call(keras$config$image_data_format, args)
}


#' Set the value of the fuzz factor used in numeric expressions.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' config_epsilon()
#' ```
#'
#' ```{r}
#' config_set_epsilon(1e-5)
#' config_epsilon()
#' ```
#'
#' ```{r}
#' # Set it back to the default value.
#' config_set_epsilon(1e-7)
#' ```
#'
#' @param value
#' float. New value of epsilon.
#'
#' @returns No return value, called for side effects.
#' @export
#' @family config backend
#' @family backend
#' @family config
#' @seealso
#' + <https://keras.io/api/utils/config_utils#setepsilon-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/set_epsilon>
#' @tether keras.config.set_epsilon
config_set_epsilon <-
function (value)
{
    args <- capture_args()
    do.call(keras$config$set_epsilon, args)
}


#' Set the default float dtype.
#'
#' @description
#'
#' # Note
#' It is not recommended to set this to `"float16"` for training,
#' as this will likely cause numeric stability issues.
#' Instead, mixed precision, which leverages
#' a mix of `float16` and `float32`. It can be configured by calling
#' `keras3::keras$mixed_precision$set_dtype_policy('mixed_float16')`.
#'
#' # Examples
#' ```{r}
#' config_floatx()
#' ```
#'
#' ```{r}
#' config_set_floatx('float64')
#' config_floatx()
#' ```
#'
#' ```{r}
#' # Set it back to float32
#' config_set_floatx('float32')
#' ```
#'
#' # Raises
#' ValueError: In case of invalid value.
#'
#' @param value
#' String; `'bfloat16'`, `'float16'`, `'float32'`, or `'float64'`.
#'
#' @returns No return value, called for side effects.
#' @export
#' @family config backend
#' @family backend
#' @family config
#' @seealso
#' + <https://keras.io/api/utils/config_utils#setfloatx-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/set_floatx>
#' @tether keras.config.set_floatx
config_set_floatx <-
function (value)
{
    keras$config$set_floatx(value)
}


#' Set the value of the image data format convention.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' config_image_data_format()
#' # 'channels_last'
#' ```
#'
#' ```{r}
#' keras3::config_set_image_data_format('channels_first')
#' config_image_data_format()
#' ```
#'
#' ```{r}
#' # Set it back to `'channels_last'`
#' keras3::config_set_image_data_format('channels_last')
#' ```
#'
#' @param data_format
#' string. `'channels_first'` or `'channels_last'`.
#'
#' @returns No return value, called for side effects.
#' @export
#' @family config backend
#' @family backend
#' @family config
#' @seealso
#' + <https://keras.io/api/utils/config_utils#setimagedataformat-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/set_image_data_format>
#' @tether keras.config.set_image_data_format
config_set_image_data_format <-
function (data_format)
{
    args <- capture_args()
    do.call(keras$config$set_image_data_format, args)
}


#' Disables safe mode globally, allowing deserialization of lambdas.
#'
#' @returns No return value, called for side effects.
#' @export
#' @family saving
#' @family config
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/enable_unsafe_deserialization>
#' @tether keras.config.enable_unsafe_deserialization
config_enable_unsafe_deserialization <-
function ()
{
    args <- capture_args()
    do.call(keras$config$enable_unsafe_deserialization, args)
}


#' Turn off interactive logging.
#'
#' @description
#' When interactive logging is disabled, Keras sends logs to `absl.logging`.
#' This is the best option when using Keras in a non-interactive
#' way, such as running a training or inference job on a server.
#'
#' @returns No return value, called for side effects.
#' @export
#' @family io utils
#' @family utils
#' @family config
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/disable_interactive_logging>
#' @tether keras.config.disable_interactive_logging
config_disable_interactive_logging <-
function ()
{
    args <- capture_args()
    do.call(keras$config$disable_interactive_logging, args)
}


#' Turn on interactive logging.
#'
#' @description
#' When interactive logging is enabled, Keras displays logs via stdout.
#' This provides the best experience when using Keras in an interactive
#' environment such as a shell or a notebook.
#'
#' @returns No return value, called for side effects.
#' @export
#' @family io utils
#' @family utils
#' @family config
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/enable_interactive_logging>
#' @tether keras.config.enable_interactive_logging
config_enable_interactive_logging <-
function ()
{
    args <- capture_args()
    do.call(keras$config$enable_interactive_logging, args)
}


#' Check if interactive logging is enabled.
#'
#' @description
#' To switch between writing logs to stdout and `absl.logging`, you may use
#' [`config_enable_interactive_logging()`] and
#' [`config_disable_interactive_logging()`].
#'
#' @returns
#' Boolean, `TRUE` if interactive logging is enabled,
#' and `FALSE` otherwise.
#'
#' @export
#' @family io utils
#' @family utils
#' @family config
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/is_interactive_logging_enabled>
#' @tether keras.config.is_interactive_logging_enabled
config_is_interactive_logging_enabled <-
function ()
{
    args <- capture_args()
    do.call(keras$config$is_interactive_logging_enabled, args)
}


#' Turn off traceback filtering.
#'
#' @description
#' Raw Keras tracebacks (also known as stack traces)
#' involve many internal frames, which can be
#' challenging to read through, while not being actionable for end users.
#' By default, Keras filters internal frames in most exceptions that it
#' raises, to keep traceback short, readable, and focused on what's
#' actionable for you (your own code).
#'
#' See also [`config_enable_traceback_filtering()`] and
#' [`config_is_traceback_filtering_enabled()`].
#'
#' If you have previously disabled traceback filtering via
#' [`config_disable_traceback_filtering()`], you can re-enable it via
#' [`config_enable_traceback_filtering()`].
#'
#' @returns No return value, called for side effects.
#' @export
#' @family traceback utils
#' @family utils
#' @family config
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/disable_traceback_filtering>
#' @tether keras.config.disable_traceback_filtering
config_disable_traceback_filtering <-
function ()
{
    args <- capture_args()
    do.call(keras$config$disable_traceback_filtering, args)
}


#' Turn on traceback filtering.
#'
#' @description
#' Raw Keras tracebacks (also known as stack traces)
#' involve many internal frames, which can be
#' challenging to read through, while not being actionable for end users.
#' By default, Keras filters internal frames in most exceptions that it
#' raises, to keep traceback short, readable, and focused on what's
#' actionable for you (your own code).
#'
#' See also [`config_disable_traceback_filtering()`] and
#' [`config_is_traceback_filtering_enabled()`].
#'
#' If you have previously disabled traceback filtering via
#' [`config_disable_traceback_filtering()`], you can re-enable it via
#' [`config_enable_traceback_filtering()`].
#'
#' @returns No return value, called for side effects.
#' @export
#' @family traceback utils
#' @family utils
#' @family config
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/enable_traceback_filtering>
#' @tether keras.config.enable_traceback_filtering
config_enable_traceback_filtering <-
function ()
{
    args <- capture_args()
    do.call(keras$config$enable_traceback_filtering, args)
}


#' Check if traceback filtering is enabled.
#'
#' @description
#' Raw Keras tracebacks (also known as stack traces)
#' involve many internal frames, which can be
#' challenging to read through, while not being actionable for end users.
#' By default, Keras filters internal frames in most exceptions that it
#' raises, to keep traceback short, readable, and focused on what's
#' actionable for you (your own code).
#'
#' See also [`config_enable_traceback_filtering()`] and
#' [`config_disable_traceback_filtering()`].
#'
#' If you have previously disabled traceback filtering via
#' [`config_disable_traceback_filtering()`], you can re-enable it via
#' [`config_enable_traceback_filtering()`].
#'
#' @returns
#' Boolean, `TRUE` if traceback filtering is enabled,
#' and `FALSE` otherwise.
#'
#' @export
#' @family traceback utils
#' @family utils
#' @family config
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/is_traceback_filtering_enabled>
#' @tether keras.config.is_traceback_filtering_enabled
config_is_traceback_filtering_enabled <-
function ()
{
    args <- capture_args()
    do.call(keras$config$is_traceback_filtering_enabled, args)
}


#' Returns the current default dtype policy object.
#'
#' @export
#' @returns A `DTypePolicy` object.
#' @tether keras.config.dtype_policy
#' @family config
#'
#  @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/dtype_policy>
config_dtype_policy <-
function ()
{
    keras$config$dtype_policy()
}

#' Sets the default dtype policy globally.
#'
#' @description
#'
#' # Examples
#' ```r
#' config_set_dtype_policy("mixed_float16")
#' ```
#' @param policy A string or `DTypePolicy` object.
#' @returns No return value, called for side effects.
#' @export
#' @family config
#' @tether keras.config.set_dtype_policy
#  @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/set_dtype_policy>
config_set_dtype_policy <-
function (policy)
{
    args <- capture_args()
    do.call(keras$config$set_dtype_policy, args)
}


#' Enable flash attention.
#'
#' @description
#' Flash attention offers performance optimization for attention layers,
#' making it especially useful for large language models (LLMs) that
#' benefit from faster and more memory-efficient attention computations.
#'
#' Once enabled, supported layers like `layer_multi_head_attention` will **attempt** to
#' use flash attention for faster computations. By default, this feature is
#' enabled.
#'
#' Note that enabling flash attention does not guarantee it will always be
#' used. Typically, the inputs must be in `float16` or `bfloat16` dtype, and
#' input layout requirements may vary depending on the backend.
#'
#'
#' @seealso [config_disable_flash_attention()] [config_is_flash_attention_enabled()]
#' @export
#' @family config
#' @tether keras.config.enable_flash_attention
#  @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/config/enable_flash_attention>
config_enable_flash_attention <-
function ()
{
  keras$config$enable_flash_attention()
}


#' Disable flash attention.
#'
#' @description
#' Flash attention offers performance optimization for attention layers,
#' making it especially useful for large language models (LLMs) that
#' benefit from faster and more memory-efficient attention computations.
#'
#' Once disabled, supported layers like `MultiHeadAttention` will not
#' use flash attention for faster computations.
#'
#' @export
#' @tether keras.config.disable_flash_attention
#' @seealso [config_is_flash_attention_enabled()] [config_enable_flash_attention()]
#' @family config
config_disable_flash_attention <-
function ()
{
  keras$config$disable_flash_attention()
}

#' Checks whether flash attention is globally enabled in Keras.
#'
#' @description
#' Flash attention is a performance-optimized method for computing attention
#' in large models, such as transformers, allowing for faster and more
#' memory-efficient operations. This function checks the global Keras
#' configuration to determine if flash attention is enabled for compatible
#' layers (e.g., `MultiHeadAttention`).
#'
#' Note that enabling flash attention does not guarantee it will always be
#' used. Typically, the inputs must be in `float16` or `bfloat16` dtype, and
#' input layout requirements may vary depending on the backend.
#'
#' @returns
#' `FALSE` if disabled; otherwise, it indicates that it is enabled.
#'
#' @export
#' @tether keras.config.is_flash_attention_enabled
#' @seealso [config_disable_flash_attention()] [config_enable_flash_attention()]
config_is_flash_attention_enabled <-
function ()
{
  keras$config$is_flash_attention_enabled()
}
