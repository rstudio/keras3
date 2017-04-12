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
#' Passing the MD5 hash will verify the file after download
#' as well as if it is already present in the cache.
#' 
#' @param fname name of the file
#' @param origin original URL of the file
#' @param untar boolean, whether the file should be decompressed
#' @param md5_hash MD5 hash of the file for verification
#' @param cache_subdir directory being used as the cache
#' 
#' @return Path to the downloaded file
#' 
#' @export
get_file <- function(fname, origin, untar = FALSE, md5_hash = NULL, cache_subdir = "datasets") {
  keras$utils$get_file(
    fname = fname,
    origin = origin,
    untar = untar,
    md5_hash = md5_hash,
    cache_subdir = cache_subdir
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
    datapath = datapath, 
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






