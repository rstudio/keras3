#' Converts a class vector (integers) to binary class matrix.
#' 
#' @details 
#' E.g. for use with [loss_categorical_crossentropy()].
#' 
#' @param y Class vector to be converted into a matrix (integers from 0 to num_classes).
#' @param num_classes Total number of classes.
#' 
#' @return A binary matrix representation of the input.
#' 
#' @export
to_categorical <- function(y, num_classes = NULL) {
  keras$utils$to_categorical(
    y = y,
    num_classes = as_nullable_integer(num_classes)
  )
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
#'   
#' @return Path to the downloaded file
#'   
#' @export
get_file <- function(fname, origin, file_hash = NULL, cache_subdir = "datasets", 
                     hash_algorithm = "auto", extract = FALSE,
                     archive_format = "auto", cache_dir = NULL) {
  keras$utils$get_file(
    fname = normalize_path(fname),
    origin = origin,
    file_hash = file_hash,
    cache_subdir = cache_subdir,
    hash_algorithm = hash_algorithm,
    extract = extract,
    archive_format = archive_format,
    cache_dir = normalize_path(cache_dir)
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
  
  if (!have_h5py())
    stop("The h5py Python package is required to read h5 files")
  
  keras$utils$HDF5Matrix(
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
#' @param axis Axis along which to normalize
#' @param order Normalization order (e.g. 2 for L2 norm) 
#' 
#' @return A normalized copy of the array.
#' 
#' @export
normalize <- function(x, axis = -1, order = 2) {
  keras$utils$normalize(
    x = x,
    axis = as.integer(axis),
    order = as.integer(order)
  )
}


#' Keras implementation
#' 
#' Obtain a reference to the Python module used for the implementation of Keras.
#' 
#' There are currently two Python modules which implement Keras:
#' 
#' - keras ("keras")
#' - tensorflow.contrib.keras ("tensorflow")
#' 
#' This function returns a reference to the implementation being currently 
#' used by the keras package. The default implementation is "tensorflow".
#' You can override this by setting the `KERAS_IMPLEMENTATION` environment
#' variable to "keras".
#' 
#' @return Reference to the Python module used for the implementation of Keras.
#' 
#' @export
implementation <- function() {
  keras
}


#' Keras backend tensor engine
#' 
#' Obtain a reference to the `keras.backend` Python module used to implement
#' tensor operations.
#'
#' @inheritParams reticulate::import
#'
#' @note See the documentation here <https://keras.io/backend/> for 
#'   additional details on the available functions.
#'
#' @return Reference to Keras backend python module.
#'  
#' @export   
backend <- function(convert = TRUE) {
  if (convert)
    keras$backend
  else
    r_to_py(keras$backend)
}


is_backend <- function(name) {
  identical(backend()$backend(), name)
}




