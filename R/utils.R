resolve_utils <- function() {
  keras$utils
}

#' Converts a class vector (integers) to binary class matrix.
#'
#' @details
#' E.g. for use with [loss_categorical_crossentropy()].
#'
#' @param y Class vector to be converted into a matrix (integers from 0 to num_classes).
#' @param num_classes Total number of classes.
#' @param dtype The data type expected by the input, as a string
#    (`float32`, `float64`, `int32`...)
#'
#' @return A binary matrix representation of the input.
#'
#' @export
to_categorical <- function(y, num_classes = NULL, dtype = "float32") {

  args <- list(
    y = y,
    num_classes = as_nullable_integer(num_classes)
  )

  if (keras_version() >= "2.2.3")
    args$dtype <- dtype

  do.call(resolve_utils()$to_categorical, args)

}


#' Downloads a file from a URL if it not already in the cache.
#'
#' Passing the MD5 hash will verify the file after download as well as if it is
#' already present in the cache.
#'
#' @param fname Name of the file. If an absolute path `/path/to/file.txt` is
#'   specified the file will be saved at that location.
#' @param origin Original URL of the file.
#' @param file_hash The expected hash string of the file after download. The
#'   sha256 and md5 hash algorithms are both supported.
#' @param cache_subdir Subdirectory under the Keras cache dir where the file is
#'   saved. If an absolute path `/path/to/folder` is specified the file will be
#'   saved at that location.
#' @param hash_algorithm Select the hash algorithm to verify the file. options
#'   are 'md5', 'sha256', and 'auto'. The default 'auto' detects the hash
#'   algorithm in use.
#' @param extract True tries extracting the file as an Archive, like tar or zip.
#' @param archive_format Archive format to try for extracting the file. Options
#'   are 'auto', 'tar', 'zip', and None. 'tar' includes tar, tar.gz, and tar.bz
#'   files. The default 'auto' is ('tar', 'zip'). None or an empty list will
#'   return no matches found.
#' @param cache_dir Location to store cached files, when `NULL` it defaults to
#'   the Keras configuration directory.
#' @param untar Deprecated in favor of 'extract'. boolean, whether the file should
#'   be decompressed
#'
#' @return Path to the downloaded file
#'
#' @export
get_file <- function(fname, origin, file_hash = NULL, cache_subdir = "datasets",
                     hash_algorithm = "auto", extract = FALSE,
                     archive_format = "auto", cache_dir = NULL,
                     untar = FALSE) {
  resolve_utils()$get_file(
    fname = normalize_path(fname),
    origin = origin,
    file_hash = file_hash,
    cache_subdir = cache_subdir,
    hash_algorithm = hash_algorithm,
    extract = extract,
    archive_format = archive_format,
    cache_dir = normalize_path(cache_dir),
    untar = untar
  )
}


#' Representation of HDF5 dataset to be used instead of an R array
#'
#' @param datapath string, path to a HDF5 file
#' @param dataset string, name of the HDF5 dataset in the file specified in datapath
#' @param start int, start of desired slice of the specified dataset
#' @param end int, end of desired slice of the specified dataset
#' @param normalizer function to be called on data when retrieved
#'
#' @return An array-like HDF5 dataset.
#'
#' @details
#' Providing `start` and `end` allows use of a slice of the dataset.
#'
#' Optionally, a normalizer function (or lambda) can be given. This will
#' be called on every slice of data retrieved.
#'
#' @export
hdf5_matrix <- function(datapath, dataset, start = 0, end = NULL, normalizer = NULL) {

  if (tensorflow::tf_version() >= "2.4")
    stop("This function have been removed in TensorFlow version 2.4 or later.")

  if (!have_h5py())
    stop("The h5py Python package is required to read h5 files")

  resolve_utils()$HDF5Matrix(
    datapath = normalize_path(datapath),
    dataset = dataset,
    start = as.integer(start),
    end = as_nullable_integer(end),
    normalizer = normalizer
  )
}

#' Normalize a matrix or nd-array
#'
#' @param x Matrix or array to normalize
#' @param axis Axis along which to normalize. Axis indexes are 1-based
#'   (pass -1 to select the last axis).
#' @param order Normalization order (e.g. 2 for L2 norm)
#'
#' @return A normalized copy of the array.
#'
#' @export
normalize <- function(x, axis = -1, order = 2) {
  resolve_utils()$normalize(
    x = x,
    axis = as_axis(axis),
    order = as.integer(order)
  )
}

#' Provide a scope with mappings of names to custom objects
#'
#' @param objects Named list of objects
#' @param expr Expression to evaluate
#'
#' @details
#' There are many elements of Keras models that can be customized with
#' user objects (e.g. losses, metrics, regularizers, etc.). When
#' loading saved models that use these functions you typically
#' need to explicitily map names to user objects via the `custom_objects`
#' parmaeter.
#'
#' The `with_custom_object_scope()` function provides an alternative that
#' lets you create a named alias for a user object that applies to an entire
#' block of code, and is automatically recognized when loading saved models.
#'
#' @examples \dontrun{
#' # define custom metric
#' metric_top_3_categorical_accuracy <-
#'   custom_metric("top_3_categorical_accuracy", function(y_true, y_pred) {
#'     metric_top_k_categorical_accuracy(y_true, y_pred, k = 3)
#'   })
#'
#' with_custom_object_scope(c(top_k_acc = sparse_top_k_cat_acc), {
#'
#'   # ...define model...
#'
#'   # compile model (refer to "top_k_acc" by name)
#'   model %>% compile(
#'     loss = "binary_crossentropy",
#'     optimizer = optimizer_nadam(),
#'     metrics = c("top_k_acc")
#'   )
#'
#'   # save the model
#'   save_model_hdf5("my_model.h5")
#'
#'   # loading the model within the custom object scope doesn't
#'   # require explicitly providing the custom_object
#'   load_model_hdf5("my_model.h5")
#' })
#' }
#'
#' @export
with_custom_object_scope <- function(objects, expr) {
  objects <- objects_with_py_function_names(objects)
  with(resolve_utils()$custom_object_scope(objects), expr)
}


objects_with_py_function_names <- function(objects) {
  if (!is.null(objects)) {
    object_names <- names(objects)
    if (is.null(object_names))
      stop("objects must be named", call. = FALSE)
    objects <- lapply(1:length(objects), function(i) {
      object <- objects[[i]]
      if (is.function(object))
        attr(object, "py_function_name") <- object_names[[i]]
      object
    })
    names(objects) <- object_names
    objects
  } else {
    NULL
  }
}


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
#' @param dtype NumPy data type (e.g. float32, float64). If this is unspecified
#'   then R doubles will be converted to the default floating point type for the
#'   current Keras backend.
#'
#' @return NumPy array with the specified `dtype` (or list of NumPy arrays if a
#'   list was passed for `x`).
#'
#' @export
keras_array <- function(x, dtype = NULL) {

  # reflect NULL
  if (is.null(x))
    return(x)

  # reflect HDF5
  if (inherits(x, "keras.utils.io_utils.HDF5Matrix"))
    return(x)

  # reflect tensor for keras v2.2 or TF implementation >= 1.12
  if (is_tensorflow_implementation()) {
    if (
      tf_version() >= "1.12" &&
      (
        is_keras_tensor(x) || is.list(x) && all(vapply(x, is_keras_tensor, logical(1)))
      )
    ) {
      return(x)
    }
  } else {
    if ((keras_version() >= "2.2.0") && is_keras_tensor(x)) {
      return(x)
    }
  }

  # error for data frames
  if (is.data.frame(x)) {
    x <- as.list(x)
  }

  # recurse for lists
  if (is.list(x))
    return(lapply(x, keras_array))

  # convert to numpy
  if (!inherits(x, "numpy.ndarray")) {

    # establish the target datatype - if we are converting a double from R
    # into numpy then use the default floatx for the current backend
    if (is.null(dtype) && is.double(x))
      dtype <- backend()$floatx()

    # convert non-array to array
    if (!is.array(x))
      x <- as.array(x)

    # do the conversion (will result in Fortran column ordering)
    x <- r_to_py(x)
  }

  # if we don't yet have a dtype then use the converted type
  if (is.null(dtype))
    dtype <- x$dtype

  # ensure we use C column ordering (won't create a new array if the array
  # is already using C ordering)
  x$astype(dtype = dtype, order = "C", copy = FALSE)
}


#' Check if Keras is Available
#'
#' Probe to see whether the Keras Python package is available in the current
#' system environment.
#'
#' @param version Minimum required version of Keras (defaults to `NULL`, no
#'   required version).
#'
#' @return Logical indicating whether Keras (or the specified minimum version of
#'   Keras) is available.
#'
#' @examples
#' \dontrun{
#' # testthat utilty for skipping tests when Keras isn't available
#' skip_if_no_keras <- function(version = NULL) {
#'   if (!is_keras_available(version))
#'     skip("Required keras version not available for testing")
#' }
#'
#' # use the function within a test
#' test_that("keras function works correctly", {
#'   skip_if_no_keras()
#'   # test code here
#' })
#' }
#'
#' @export
is_keras_available <- function(version = NULL) {
  implementation_module <- resolve_implementation_module()
  if (reticulate::py_module_available(implementation_module)) {
    if (!is.null(version))
      keras_version() >= version
    else
      TRUE
  } else {
    FALSE
  }
}


#' Keras implementation
#'
#' Obtain a reference to the Python module used for the implementation of Keras.
#'
#' There are currently two Python modules which implement Keras:
#'
#' - keras ("keras")
#' - tensorflow.keras ("tensorflow")
#'
#' This function returns a reference to the implementation being currently
#' used by the keras package. The default implementation is "keras".
#' You can override this by setting the `KERAS_IMPLEMENTATION` environment
#' variable to "tensorflow".
#'
#' @return Reference to the Python module used for the implementation of Keras.
#'
#' @export
implementation <- function() {
  keras
}


is_backend <- function(name) {
  identical(backend()$backend(), name)
}

is_windows <- function() {
  identical(.Platform$OS.type, "windows")
}

is_osx <- function() {
  Sys.info()["sysname"] == "Darwin"
}

is_layer <- function(object) {
  inherits(object, "keras.engine.topology.Layer")
}

relative_to <- function(dir, file) {

  # normalize paths
  dir <- normalizePath(dir, mustWork = FALSE, winslash = "/")
  file <- normalizePath(file, mustWork = FALSE, winslash = "/")

  # ensure directory ends with a /
  if (!identical(substr(dir, nchar(dir), nchar(dir)), "/")) {
    dir <- paste(dir, "/", sep="")
  }

  # if the file is prefixed with the directory, return a relative path
  if (identical(substr(file, 1, nchar(dir)), dir))
    file <- substr(file, nchar(dir) + 1, nchar(file))

  # simplify ./
  if (identical(substr(file, 1, 2), "./"))
    file <- substr(file, 3, nchar(file))

  file
}


is_keras_tensor <- function(x) {
  if (is_tensorflow_implementation()) {
    if (tensorflow::tf_version() >= "2.0") tensorflow::tf$is_tensor(x) else tensorflow::tf$contrib$framework$is_tensor(x)
  } else {
    k_is_tensor(x)
  }
}



assert_all_dots_named <- function(envir = parent.frame(), cl) {

  x <- eval(quote(list(...)), envir)
  if(!length(x))
    return()

  x <- names(x)
  if(is.character(x) && !anyNA(x) && all(x != ""))
    return()

  stop("All arguments provided to `...` must be named.\n",
       "Call with unnamed arguments in dots:\n  ",
       paste(deparse(cl, 500L), collapse = "\n"))
}

capture_args <- function(cl, modifiers = NULL, ignore = NULL,
                         envir = parent.frame(), fn = sys.function(-1)) {

  ## bug: match.call() resolves incorrectly if dots are from not the default sys.parent()
  ## e.g, this fails if dots originate from the callers caller:
  #    cl <- eval(quote(match.call()), parent.frame())
  ## workaround: caller must call match.call() from the correct frame

  ## note: capture_args() must always be called at the top level of the intended function body.
  ## sys.function(-1) resolves to the incorrect function if the  capture_args()
  ## call is itself a promise in another call. E.g.,:
  ##   do.call(foo, capture_args(match.call())) fails because fn resolves to do.call()

  fn_arg_nms <- names(formals(fn))
  known_args <- intersect(names(cl), fn_arg_nms)
  known_args <- setdiff(known_args, ignore)
  names(known_args) <- known_args
  cl2 <- c(quote(list), lapply(known_args, as.symbol))

  if("..." %in% fn_arg_nms && !"..." %in% ignore) {
    assert_all_dots_named(envir, cl)
    # this might reorder args by assuming ... are last, but it doesn't matter
    # since everything is supplied as a keyword arg to the Python side anyway
    cl2 <- c(cl2, quote(...))
  }

  args <- eval(as.call(cl2), envir)

  for(nm in intersect(names(args), ignore))
    args[[nm]] <- NULL

  nms_to_modify <- intersect(names(args), names(modifiers))
  for (nm in nms_to_modify)
    args[nm] <- list(modifiers[[nm]](args[[nm]]))
   # list() so if modifier returns NULL, don't remove the arg

  args
}

# TODO
plot_model <- function(...) {}
