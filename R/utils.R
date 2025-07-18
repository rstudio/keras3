


#' Resets all state generated by Keras.
#'
#' @description
#' Keras manages a global state, which it uses to implement the Functional
#' model-building API and to uniquify autogenerated layer names.
#'
#' If you are creating many models in a loop, this global state will consume
#' an increasing amount of memory over time, and you may want to clear it.
#' Calling `clear_session()` releases the global state: this helps avoid
#' clutter from old models and layers, especially when memory is limited.
#'
#' Example 1: calling `clear_session()` when creating models in a loop
#'
#' ```{r}
#' for (i in 1:100) {
#'   # Without `clear_session()`, each iteration of this loop will
#'   # slightly increase the size of the global state managed by Keras
#'   model <- keras_model_sequential()
#'   for (j in 1:10) {
#'     model <- model |> layer_dense(units = 10)
#'   }
#' }
#'
#' for (i in 1:100) {
#'   # With `clear_session()` called at the beginning,
#'   # Keras starts with a blank state at each iteration
#'   # and memory consumption is constant over time.
#'   clear_session()
#'   model <- keras_model_sequential()
#'   for (j in 1:10) {
#'     model <- model |> layer_dense(units = 10)
#'   }
#' }
#' ```
#'
#' Example 2: resetting the layer name generation counter
#'
#' ```{r, include = FALSE}
#' clear_session()
#' ```
#'
#'
#' ```{r}
#' layers <- lapply(1:10, \(i) layer_dense(units = 10))
#'
#' new_layer <- layer_dense(units = 10)
#' print(new_layer$name)
#'
#' clear_session()
#' new_layer <- layer_dense(units = 10)
#' print(new_layer$name)
#' ```
#'
#' @param free_memory
#' Whether to call Python garbage collection.
#' It's usually a good practice to call it to make sure
#' memory used by deleted objects is immediately freed.
#' However, it may take a few seconds to execute, so
#' when using `clear_session()` in a short loop,
#' you may want to skip it.
#'
#' @returns `NULL`, invisibly, called for side effects.
#' @export
#' @family backend
#' @family utils
#' @seealso
#' + <https://keras.io/api/utils/config_utils#clearsession-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/clear_session>
#' @tether keras.utils.clear_session
clear_session <-
function (free_memory = TRUE)
{
    args <- capture_args()
    do.call(keras$utils$clear_session, args)
}



#' Returns the list of input tensors necessary to compute `tensor`.
#'
#' @description
#' Output will always be a list of tensors
#' (potentially with 1 element).
#'
#' # Example
#'
#' ```{r}
#' input <- keras_input(c(3))
#' output <- input |> layer_dense(4) |> op_multiply(5)
#' reticulate::py_id(get_source_inputs(output)[[1]]) ==
#' reticulate::py_id(input)
#' ```
#'
#' @returns
#' List of input tensors.
#'
#' @param tensor
#' The tensor to start from.
#'
#' @export
#' @family utils
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_source_inputs>
#' @tether keras.utils.get_source_inputs
get_source_inputs <-
function (tensor)
{
    keras$utils$get_source_inputs(tensor)
}


#' Downloads a file from a URL if it not already in the cache.
#'
#' @description
#' By default the file at the url `origin` is downloaded to the
#' cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
#' and given the filename `fname`. The final location of a file
#' `example.txt` would therefore be `~/.keras/datasets/example.txt`.
#' Files in `.tar`, `.tar.gz`, `.tar.bz`, and `.zip` formats can
#' also be extracted.
#'
#' Passing a hash will verify the file after download. The command line
#' programs `shasum` and `sha256sum` can compute the hash.
#'
#' # Examples
#' ```{r}
#' path_to_downloaded_file <- get_file(
#'     "flower_photos",
#'     origin = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
#'     extract = TRUE
#' )
#' ```
#'
#' @returns
#' Path to the downloaded file.
#'
#' ** Warning on malicious downloads **
#'
#' Downloading something from the Internet carries a risk.
#' NEVER download a file/archive if you do not trust the source.
#' We recommend that you specify the `file_hash` argument
#' (if the hash of the source file is known) to make sure that the file you
#' are getting is the one you expect.
#'
#' @param fname
#' If the target is a single file, this is your desired
#' local name for the file.
#' If `NULL`, the name of the file at `origin` will be used.
#' If downloading and extracting a directory archive,
#' the provided `fname` will be used as extraction directory
#' name (only if it doesn't have an extension).
#'
#' @param origin
#' Original URL of the file.
#'
#  @param untar
#  Deprecated in favor of `extract` argument.
#  Boolean, whether the file is a tar archive that should
#  be extracted.
#
#  @param md5_hash
#  Deprecated in favor of `file_hash` argument.
#  md5 hash of the file for file integrity verification.
#'
#' @param file_hash
#' The expected hash string of the file after download.
#' The sha256 and md5 hash algorithms are both supported.
#'
#' @param cache_subdir
#' Subdirectory under the Keras cache dir where the file is
#' saved. If an absolute path, e.g. `"/path/to/folder"` is
#' specified, the file will be saved at that location.
#'
#' @param hash_algorithm
#' Select the hash algorithm to verify the file.
#' options are `"md5'`, `"sha256'`, and `"auto'`.
#' The default 'auto' detects the hash algorithm in use.
#'
#' @param extract
#' If `TRUE`, extracts the archive. Only applicable to compressed
#' archive files like tar or zip.
#'
#' @param archive_format
#' Archive format to try for extracting the file.
#' Options are `"auto'`, `"tar'`, `"zip'`, and `NULL`.
#' `"tar"` includes tar, tar.gz, and tar.bz files.
#' The default `"auto"` corresponds to `c("tar", "zip")`.
#' `NULL` or an empty list will return no matches found.
#'
#' @param cache_dir
#' Location to store cached files, when `NULL` it
#' defaults to `Sys.getenv("KERAS_HOME", "~/.keras/")`.
#'
#' @param force_download
#' If `TRUE`, the file will always be re-downloaded
#' regardless of the cache state.
#'
#' @param ... For forward/backward compatability.
#'
#' @export
#' @family utils
#' @seealso
#' + <https://keras.io/api/utils/python_utils#getfile-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file>
#' @tether keras.utils.get_file
get_file <-
function (fname = NULL, origin = NULL, ...,
    file_hash = NULL, cache_subdir = "datasets", hash_algorithm = "auto",
    extract = FALSE, archive_format = "auto", cache_dir = NULL,
    force_download = FALSE)
{
    args <- capture_args()
    do.call(keras$utils$get_file, args)
}



#  Convert a Keras model to dot format.
#
#  @returns
#  A `pydot.Dot` instance representing the Keras model or
#  a `pydot.Cluster` instance representing nested model if
#  `subgraph=TRUE`.
#
#  @param model
#  A Keras model instance.
#
#  @param show_shapes
#  whether to display shape information.
#
#  @param show_dtype
#  whether to display layer dtypes.
#
#  @param show_layer_names
#  whether to display layer names.
#
#  @param rankdir
#  `rankdir` argument passed to PyDot,
#  a string specifying the format of the plot: `"TB"`
#  creates a vertical plot; `"LR"` creates a horizontal plot.
#
#  @param expand_nested
#  whether to expand nested Functional models
#  into clusters.
#
#  @param dpi
#  Image resolution in dots per inch.
#
#  @param subgraph
#  whether to return a `pydot.Cluster` instance.
#
#  @param show_layer_activations
#  Display layer activations (only for layers that
#  have an `activation` property).
#
#  @param show_trainable
#  whether to display if a layer is trainable.
#
#  @param ...
#  For forward/backward compatability.
#
# @export
#  @noRd
#  @family utils
#  @seealso
#  + <https://keras.io/api/utils/model_plotting_utils#modeltodot-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/model_to_dot>
#  @tether keras.utils.model_to_dot
# model_to_dot <-
function (model, show_shapes = FALSE, show_dtype = FALSE, show_layer_names = TRUE,
    rankdir = "TB", expand_nested = FALSE, dpi = 200L, subgraph = FALSE,
    show_layer_activations = FALSE, show_trainable = FALSE, ...)
{
    args <- capture_args(list(dpi = as_integer))
    do.call(keras$utils$model_to_dot, args)
}


#' Normalizes an array.
#'
#' @description
#' If the input is an R array, an R array will be returned.
#' If it's a backend tensor, a backend tensor will be returned.
#'
#' @returns
#' A normalized copy of the array.
#'
#' @param x
#' Array to normalize.
#'
#' @param axis
#' axis along which to normalize.
#'
#' @param order
#' Normalization order (e.g. `order=2` for L2 norm).
#'
#' @export
#' @family numerical utils
#' @family utils
#' @seealso
#' + <https://keras.io/api/utils/python_utils#normalize-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/normalize>
#' @tether keras.utils.normalize
normalize <-
function (x, axis = -1L, order = 2L)
{
    args <- capture_args(list(axis = as_axis, order = as_integer))
    do.call(keras$utils$normalize, args)
}


#' Converts a class vector (integers) to binary class matrix.
#'
#' @description
#' E.g. for use with [`loss_categorical_crossentropy()`].
#'
#' # Examples
#' ```{r}
#' a <- to_categorical(c(0, 1, 2, 3), num_classes=4)
#' print(a)
#' ```
#'
#' ```{r}
#' b <- array(c(.9, .04, .03, .03,
#'               .3, .45, .15, .13,
#'               .04, .01, .94, .05,
#'               .12, .21, .5, .17),
#'               dim = c(4, 4))
#' loss <- op_categorical_crossentropy(a, b)
#' loss
#' ```
#'
#' ```{r}
#' loss <- op_categorical_crossentropy(a, a)
#' loss
#' ```
#'
#' @returns
#' A binary matrix representation of the input as an R array. The class
#' axis is placed last.
#'
#' @param x
#' Array-like with class values to be converted into a matrix
#' (integers from 0 to `num_classes - 1`).
#' R factors are coerced to integer and offset to be 0-based, i.e.,
#' `as.integer(x) - 1L`.
#'
#' @param num_classes
#' Total number of classes. If `NULL`, this would be inferred
#' as `max(x) + 1`. Defaults to `NULL`.
#'
#' @export
#' @family numerical utils
#' @family utils
#' @seealso
#' + [`op_one_hot()`], which does the same operation as `to_categorical()`, but
#'   operating on tensors.
#' + [`loss_sparse_categorical_crossentropy()`], which can
#'   accept labels (`y_true`) as an integer vector, instead of as a dense one-hot
#'   matrix.
#' + <https://keras.io/api/utils/python_utils#tocategorical-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical>
#'
#' @tether keras.utils.to_categorical
to_categorical <-
function (x, num_classes = NULL)
{
  if (inherits(x, "factor")) {
    x <- array(as.integer(x) - 1L, dim = dim(x) %||% length(x))
    if (is.null(num_classes))
      num_classes <- length(levels(x))
  }

  x <- as_integer_array(x)
  num_classes <- as_integer(num_classes)
  keras$utils$to_categorical(x, num_classes)
}



#' Sets all random seeds (Python, NumPy, and backend framework, e.g. TF).
#'
#' @description
#' You can use this utility to make almost any Keras program fully
#' deterministic. Some limitations apply in cases where network communications
#' are involved (e.g. parameter server distribution), which creates additional
#' sources of randomness, or when certain non-deterministic cuDNN ops are
#' involved.
#'
#' This sets:
#' - the R session seed: [`set.seed()`]
#' - the Python session seed: `import random; random.seed(seed)`
#' - the Python NumPy seed: `import numpy; numpy.random.seed(seed)`
#' - the TensorFlow seed: `tf$random$set_seed(seed)` (only if TF is installed)
#' - The Torch seed: `import("torch")$manual_seed(seed)` (only if the backend is torch)
#' - and disables Python hash randomization.
#'
#' Note that the TensorFlow seed is set even if you're not using TensorFlow
#' as your backend framework, since many workflows leverage `tf$data`
#' pipelines (which feature random shuffling). Likewise many workflows
#' might leverage NumPy APIs.
#'
#' @param seed
#' Integer, the random seed to use.
#'
#' @returns No return value, called for side effects.
#' @export
#' @family utils
#' @seealso
#' + <https://keras.io/api/utils/python_utils#setrandomseed-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed>
#'
#' @tether keras.utils.set_random_seed
set_random_seed <-
function (seed)
{
    seed <- as_integer(seed)
    set.seed(seed)
    reticulate::py_set_seed(seed)
    keras$utils$set_random_seed(seed)
}


#' Pads sequences to the same length.
#'
#' @description
#' This function transforms a list (of length `num_samples`)
#' of sequences (lists of integers)
#' into a 2D NumPy array of shape `(num_samples, num_timesteps)`.
#' `num_timesteps` is either the `maxlen` argument if provided,
#' or the length of the longest sequence in the list.
#'
#' Sequences that are shorter than `num_timesteps`
#' are padded with `value` until they are `num_timesteps` long.
#'
#' Sequences longer than `num_timesteps` are truncated
#' so that they fit the desired length.
#'
#' The position where padding or truncation happens is determined by
#' the arguments `padding` and `truncating`, respectively.
#' Pre-padding or removing values from the beginning of the sequence is the
#' default.
#'
#' ```{r}
#' sequence <- list(c(1), c(2, 3), c(4, 5, 6))
#' pad_sequences(sequence)
#' ```
#'
#' ```{r}
#' pad_sequences(sequence, value=-1)
#' ```
#'
#' ```{r}
#' pad_sequences(sequence, padding='post')
#' ```
#'
#' ```{r}
#' pad_sequences(sequence, maxlen=2)
#' ```
#'
#' @returns
#' Array with shape `(len(sequences), maxlen)`
#'
#' @param sequences
#' List of sequences (each sequence is a list of integers).
#'
#' @param maxlen
#' Optional Int, maximum length of all sequences. If not provided,
#' sequences will be padded to the length of the longest individual
#' sequence.
#'
#' @param dtype
#' (Optional, defaults to `"int32"`). Type of the output sequences.
#' To pad sequences with variable length strings, you can use `object`.
#'
#' @param padding
#' String, "pre" or "post" (optional, defaults to `"pre"`):
#' pad either before or after each sequence.
#'
#' @param truncating
#' String, "pre" or "post" (optional, defaults to `"pre"`):
#' remove values from sequences larger than
#' `maxlen`, either at the beginning or at the end of the sequences.
#'
#' @param value
#' Float or String, padding value. (Optional, defaults to `0`)
#'
#' @export
#' @family utils
#' @seealso
#' + <https://keras.io/api/data_loading/timeseries#padsequences-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences>
#'
#' @tether keras.utils.pad_sequences
pad_sequences <-
function (sequences, maxlen = NULL, dtype = "int32", padding = "pre",
    truncating = "pre", value = 0)
{
    args <- capture_args(list(maxlen = as_integer, sequences = function (x)
    lapply(x, as.list)))
    do.call(keras$utils$pad_sequences, args)
}


# --------------------------------------------------------------------------------




#' Keras array object
#'
#' Convert an R vector, matrix, or array object to an array that has the optimal
#' in-memory layout and floating point data type for the current Keras backend.
#'
#' Keras does frequent row-oriented access to arrays (for shuffling and drawing
#' batches) so the order of arrays created by this function is always
#' row-oriented ("C" as opposed to "Fortran" ordering, which is the default for
#' R arrays).
#'
#' If the passed array is already a NumPy array with the desired `dtype` and "C"
#' order then it is returned unmodified (no additional copies are made).
#'
#' @param x Object or list of objects to convert
#' @param dtype NumPy data type (e.g. `"float32"`, `"float64"`). If this is
#'   unspecified then R doubles will be converted to the default floating point
#'   type for the current Keras backend.
#'
#' @returns NumPy array with the specified `dtype` (or list of NumPy arrays if a
#'   list was passed for `x`).
#'
#' @keywords internal
#' @noRd
keras_array <- function(x, dtype = NULL) {

  # reflect NULL
  if (is.null(x))
    return(x)

  # reflect tensors
  # allow passing things like pandas.Series(), for workarounds like
  # https://github.com/rstudio/keras/issues/1341
  if (is_py_object(x))
    return(x)

  # recurse for lists/data.frames
  if (is.list(x))
    return(lapply(x, keras_array))

  # establish the target datatype - if we are converting a double from R
  # into numpy then use the default floatx for the current backend
  if (is.null(dtype) && is.double(x))
    dtype <- config_floatx()

  np_array(x, dtype)
}






#' Plot a Keras model
#'
#' @param x A Keras model instance
#' @param to_file File name of the plot image. If `NULL` (the default), the
#'   model is drawn on the default graphics device. Otherwise, a file is saved.
#' @param show_shapes whether to display shape information.
#' @param show_dtype whether to display layer dtypes.
#' @param show_layer_names whether to display layer names.
#' @param ... passed on to Python `keras.utils.model_to_dot()`. Used for forward and
#'   backward compatibility.
#' @param rankdir a string specifying the format of the plot: `'TB'` creates a
#'   vertical plot; `'LR'` creates a horizontal plot. (argument passed to PyDot)
#' @param expand_nested Whether to expand nested models into clusters.
#' @param dpi Dots per inch. Increase this value if the image text appears
#'   excessively pixelated.
#' @param layer_range `list` containing two character strings, which is the
#'   starting layer name and ending layer name (both inclusive) indicating the
#'   range of layers for which the plot will be generated. It also accepts regex
#'   patterns instead of exact name. In such case, start predicate will be the
#'   first element it matches to `layer_range[1]` and the end predicate will be
#'   the last element it matches to `layer_range[2]`. By default `NULL` which
#'   considers all layers of model. Note that you must pass range such that the
#'   resultant subgraph must be complete.
#' @param show_layer_activations Display layer activations (only for layers that
#'   have an `activation` property).
#' @param show_trainable
#' whether to display if a layer is trainable.
#'
#' @returns Nothing, called for it side effects.
#'
#' @section Raises: ValueError: if `plot(model)` is called before the model is
#'   built, unless an `input_shape = ` argument was supplied to
#'   `keras_model_sequential()`.
#'
#' @section Requirements:
#'   This function requires pydot and graphviz.
#'
#'   `pydot` is by default installed by `install_keras()`, but if you installed
#'   Keras by other means, you can install `pydot` directly with:
#'   ````r
#'   reticulate::py_install("pydot", pip = TRUE)
#'   ````
#'   You can install graphviz directly from here:
#'   <https://graphviz.gitlab.io/download/>
#'
#'   On most Linux platforms, can install graphviz via the package manager.
#'   For example, on Ubuntu/Debian you can install with
#'   ```sh
#'   sudo apt install graphviz
#'   ```
#'   On macOS you can install graphviz using `brew`:
#'   ```sh
#'   brew install graphviz
#'   ```
#'   In a conda environment, you can install graphviz with:
#'   ```r
#'   reticulate::conda_install(packages = "graphviz")
#'   # Restart the R session after install.
#'   ```
#' @tether keras.utils.model_to_dot
#' @export
plot.keras.src.models.model.Model <-
function(x,
         show_shapes = FALSE,
         show_dtype = FALSE,
         show_layer_names = FALSE,
         ...,
         rankdir = "TB",
         expand_nested = FALSE,
         dpi = 200,
         layer_range = NULL,
         show_layer_activations = FALSE,
         show_trainable = NA,
         to_file = NULL) {

  args <- capture_args(ignore = c("x", "to_file", "show_trainable"),
                        force = c("show_layer_names"))
  args$model <- x

  if (is.na(show_trainable)) {
    built <- as_r_value(py_get_attr(x, "built", silent = TRUE)) %||% FALSE
    show_trainable <- built && as.logical(length(x$non_trainable_weights))
  }
  args$show_trainable <- show_trainable

  if (is.null(to_file)) {

    if (isTRUE(getOption('knitr.in.progress'))) {

      options <- knitr::opts_current$get()
      plot_counter <- asNamespace("knitr")$plot_counter
      number <- plot_counter()

      file <- knitr::fig_path(
        suffix  = options$dev %||% ".png",
        options = options,
        number  = number
      )

      dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
      # args$dpi <- args$dpi %||% options$dpi

    } else {

      file <- tempfile(paste0("keras_", x$name), fileext = ".png")
      on.exit(unlink(file), add = TRUE)

    }
  } else {
    file <- to_file
  }

  tryCatch({
    dot <- do.call(keras$utils$model_to_dot, args)
    dot$write(file, format = tools::file_ext(file))
    },
    error = function(e) {
      message("See ?keras3::plot.keras.src.models.model.Model for",
              " instructions on how to install graphviz and pydot.")
      e$call <- sys.call(1)
      stop(e)
    }
  )

  if (!is.null(to_file))
    return(invisible())

  if (isTRUE(getOption('knitr.in.progress')))
    return(knitr::include_graphics(file, dpi = dpi))

  img <- png::readPNG(file, native = TRUE)
  graphics::plot.new()
  graphics::plot.window(xlim = c(0, ncol(img)), ylim = c(0, nrow(img)),
                        asp = 1, yaxs = "i", xaxs = "i")
  graphics::rasterImage(img, 0, 0, ncol(img), nrow(img), interpolate = TRUE)
  invisible()
}


#' Zip lists
#'
#' This is conceptually similar to `zip()` in Python, or R functions
#' `purrr::transpose()` and `data.table::transpose()` (albeit, accepting
#' elements in `...` instead of a single list), with one crucial difference: if
#' the provided objects are named, then matching is done by names, not
#' positions.
#'
#' All arguments supplied must be of the same length. If positional matching is
#' required, then all arguments provided must be unnamed. If matching by names,
#' then all arguments must have the same set of names, but they can be in
#' different orders.
#'
#' @param ... R lists or atomic vectors, optionally named.
#'
#' @returns A inverted list
#' @export
#'
#' @family data utils
#' @family utils
#'
#' @examples
#' gradients <- list("grad_for_wt_1", "grad_for_wt_2", "grad_for_wt_3")
#' weights <- list("weight_1", "weight_2", "weight_3")
#' str(zip_lists(gradients, weights))
#' str(zip_lists(gradient = gradients, weight = weights))
#'
#' names(gradients) <- names(weights) <- paste0("layer_", 1:3)
#' str(zip_lists(gradients, weights[c(3, 1, 2)]))
#'
#' names(gradients) <- paste0("gradient_", 1:3)
#' try(zip_lists(gradients, weights)) # error, names don't match
#' # call unname directly for positional matching
#' str(zip_lists(unname(gradients), unname(weights)))
zip_lists <- function(...) {
  dots <- list(...)
  if(length(dots) == 1)
    dots <- dots[[1L]]

  nms1 <- names(dots[[1L]])

  if (is.character(nms1))
    if (!anyDuplicated(nms1) && !anyNA(nms1) && !all(nzchar(nms1)))
      stop("All names must be unique. Call `unname()` if you want positional matching")

  for(i in seq_along(dots)[-1L]) {
    d_nms <- names(dots[[i]])
    if(identical(nms1, d_nms))
       next
    if(setequal(nms1, d_nms)) {
      dots[[i]] <- dots[[i]][nms1]
      next
    }
    stop("All names of arguments provided to `zip_lists()` must match.",
         " Call `unname()` on each argument if you want positional matching")
  }

  ans <- .mapply(list, dots, NULL)
  names(ans) <- nms1
  ans
}



#' Generate a Random Array
#'
#' This function generates an R array with random numbers.
#' The dimensions of the array are specified by the user.
#' The generation function for the random numbers can also be customized.
#'
#' @param ... Dimensions for the array as separate integers or as a single vector.
#' @param gen A function for generating random numbers, defaulting to `runif`.
#'
#' @returns Returns an array with the specified dimensions filled with random numbers.
#' @noRd
#'
#' @examples
#' # Create a 3x3 matrix with random numbers from uniform distribution
#' random_array(3, 3)
#'
#' # Create a 2x2x2 array with random numbers from normal distribution
#' random_array(2, 2, 2, gen = rnorm)
#'
#' # Create a 2x2 array with a sequence of integers.
#' random_array(2, 2, gen = seq)
#'
#' @keywords internal
random_array <- function(..., gen = stats::runif) {
  dim <- unlist(c(...), use.names = FALSE)
  array(gen(prod(dim)), dim = dim)
}
