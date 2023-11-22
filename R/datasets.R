


#' CIFAR10 small image classification
#'
#' Dataset of 50,000 32x32 color training images, labeled over 10 categories,
#' and 10,000 test images.
#'
#' @return Lists of training and test data: `train$x, train$y, test$x, test$y`.
#'
#' The `x` data is an array of RGB image data with shape (num_samples, 3, 32,
#' 32).
#'
#' The `y` data is an array of category labels (integers in range 0-9) with
#' shape (num_samples).
#'
#' @family datasets
#'
#' @export
dataset_cifar10 <- function() {
  dataset <- keras$datasets$cifar10$load_data()
  as_dataset_list(dataset)
}



#' CIFAR100 small image classification
#'
#' Dataset of 50,000 32x32 color training images, labeled over 100 categories,
#' and 10,000 test images.
#'
#' @param label_mode one of "fine", "coarse".
#'
#' @return Lists of training and test data: `train$x, train$y, test$x, test$y`.
#'
#' The `x` data is an array of RGB image data with shape (num_samples, 3, 32, 32).
#'
#' The `y` data is an array of category labels with shape (num_samples).
#'
#' @family datasets
#'
#' @export
dataset_cifar100 <- function(label_mode = c("fine", "coarse")) {
  dataset <- keras$datasets$cifar100$load_data(
    label_mode = match.arg(label_mode)
  )
  as_dataset_list(dataset)
}



#' IMDB Movie reviews sentiment classification
#'
#' Dataset of 25,000 movies reviews from IMDB, labeled by sentiment
#' (positive/negative). Reviews have been preprocessed, and each review is
#' encoded as a sequence of word indexes (integers). For convenience, words are
#' indexed by overall frequency in the dataset, so that for instance the integer
#' "3" encodes the 3rd most frequent word in the data. This allows for quick
#' filtering operations such as: "only consider the top 10,000 most common
#' words, but eliminate the top 20 most common words".
#'
#' As a convention, "0" does not stand for a specific word, but instead is used
#' to encode any unknown word.
#'
#' @param path Where to cache the data (relative to `~/.keras/dataset`).
#' @param num_words Max number of words to include. Words are ranked by how
#'   often they occur (in the training set) and only the most frequent words are
#'   kept
#' @param skip_top Skip the top N most frequently occuring words (which may not
#'   be informative).
#' @param maxlen sequences longer than this will be filtered out.
#' @param seed random seed for sample shuffling.
#' @param start_char The start of a sequence will be marked with this character.
#'   Set to 1 because 0 is usually the padding character.
#' @param oov_char Words that were cut out because of the `num_words` or
#'   `skip_top` limit will be replaced with this character.
#' @param index_from Index actual words with this index and higher.
#'
#' @return Lists of training and test data: `train$x, train$y, test$x, test$y`.
#'
#'   The `x` data includes integer sequences. If the `num_words` argument was
#'   specific, the maximum possible index value is `num_words-1`. If the
#'   `maxlen` argument was specified, the largest possible sequence length is
#'   `maxlen`.
#'
#'   The `y` data includes a set of integer labels (0 or 1).
#'
#'   The `dataset_imdb_word_index()` function returns a list where the
#'   names are words and the values are integer.
#'
#' @family datasets
#'
#' @export
dataset_imdb <- function(path = "imdb.npz", num_words = NULL, skip_top = 0L, maxlen = NULL,
                         seed = 113L, start_char = 1L, oov_char = 2L, index_from = 3L) {
  dataset <- keras$datasets$imdb$load_data(
    path = path,
    num_words = as_nullable_integer(num_words),
    skip_top = as.integer(skip_top),
    maxlen = as_nullable_integer(maxlen),
    seed = as.integer(seed),
    start_char = as.integer(start_char),
    oov_char = as.integer(oov_char),
    index_from = as.integer(index_from)
  )

  as_sequences_dataset_list(dataset)

}


#' @rdname dataset_imdb
#' @export
dataset_imdb_word_index <- function(path = "imdb_word_index.json") {
  keras$datasets$imdb$get_word_index(path)
}


#' Reuters newswire topics classification
#'
#' Dataset of 11,228 newswires from Reuters, labeled over 46 topics. As with
#' [dataset_imdb()] , each wire is encoded as a sequence of word indexes (same
#' conventions).
#'
#' @param path Where to cache the data (relative to `~/.keras/dataset`).
#' @param num_words Max number of words to include. Words are ranked by how
#'   often they occur (in the training set) and only the most frequent words are
#'   kept
#' @param skip_top Skip the top N most frequently occuring words (which may not
#'   be informative).
#' @param maxlen Truncate sequences after this length.
#' @param test_split Fraction of the dataset to be used as test data.
#' @param seed Random seed for sample shuffling.
#' @param start_char The start of a sequence will be marked with this character.
#'   Set to 1 because 0 is usually the padding character.
#' @param oov_char words that were cut out because of the `num_words` or
#'   `skip_top` limit will be replaced with this character.
#' @param index_from index actual words with this index and higher.
#'
#' @return Lists of training and test data: `train$x, train$y, test$x, test$y`
#'   with same format as [dataset_imdb()]. The `dataset_reuters_word_index()`
#'   function returns a list where the names are words and the values are
#'   integer. e.g. `word_index[["giraffe"]]` might return `1234`.
#'
#' @family datasets
#'
#' @export
dataset_reuters <- function(path = "reuters.npz", num_words = NULL, skip_top = 0L, maxlen = NULL,
                            test_split = 0.2, seed = 113L, start_char = 1L, oov_char = 2L,
                            index_from = 3L) {
  dataset <- keras$datasets$reuters$load_data(
    path = path,
    num_words = as_nullable_integer(num_words),
    skip_top = as.integer(skip_top),
    maxlen = as_nullable_integer(maxlen),
    test_split = test_split,
    seed = as.integer(seed),
    start_char = as.integer(start_char),
    oov_char = as.integer(oov_char),
    index_from = as.integer(index_from)
  )
  as_sequences_dataset_list(dataset)
}


#' @rdname dataset_reuters
#' @export
dataset_reuters_word_index <- function(path = "reuters_word_index.pkl") {
  keras$datasets$reuters$get_word_index(path = path)
}



#' MNIST database of handwritten digits
#'
#' Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
#'
#' @param path Path where to cache the dataset locally (relative to ~/.keras/datasets).
#'
#' @return Lists of training and test data: `train$x, train$y, test$x, test$y`, where
#'   `x` is an array of grayscale image data with shape (num_samples, 28, 28) and `y`
#'   is an array of digit labels (integers in range 0-9) with shape (num_samples).
#'
#' @family datasets
#'
#' @export
dataset_mnist <- function(path = "mnist.npz") {
  dataset <- keras$datasets$mnist$load_data(path)
  as_dataset_list(dataset)
}


#' Fashion-MNIST database of fashion articles
#'
#' Dataset of 60,000 28x28 grayscale images of the 10 fashion article classes,
#' along with a test set of 10,000 images. This dataset can be used as a drop-in
#' replacement for MNIST. The class labels are encoded as integers from 0-9 which
#' correspond to T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt,
# 'Sneaker, Bag and Ankle boot.
#'
#' @return Lists of training and test data: `train$x, train$y, test$x, test$y`, where
#'   `x` is an array of grayscale image data with shape (num_samples, 28, 28) and `y`
#'   is an array of article labels (integers in range 0-9) with shape (num_samples).
#'
#' @details Dataset of 60,000 28x28 grayscale images of 10 fashion categories,
#' along with a test set of 10,000 images. This dataset can be used as a drop-in
#' replacement for MNIST. The class labels are:
#'
#' * 0 - T-shirt/top
#' * 1 - Trouser
#' * 2 - Pullover
#' * 3 - Dress
#' * 4 - Coat
#' * 5 - Sandal
#' * 6 - Shirt
#' * 7 - Sneaker
#' * 8 - Bag
#' * 9 - Ankle boot
#'
#' @family datasets
#'
#' @export
dataset_fashion_mnist <- function() {
  dataset <- keras$datasets$fashion_mnist$load_data()
  as_dataset_list(dataset)
}


#' Boston housing price regression dataset
#'
#' Dataset taken from the StatLib library which is maintained at Carnegie Mellon
#' University.
#'
#' @param path Path where to cache the dataset locally (relative to
#'   ~/.keras/datasets).
#' @param test_split fraction of the data to reserve as test set.
#' @param seed Random seed for shuffling the data before computing the test
#'   split.
#'
#' @return Lists of training and test data: `train$x, train$y, test$x, test$y`.
#'
#' Samples contain 13 attributes of houses at different locations around
#' the Boston suburbs in the late 1970s. Targets are the median values of the
#' houses at a location (in k$).
#'
#' @family datasets
#'
#' @export
dataset_boston_housing <- function(path = "boston_housing.npz", test_split = 0.2, seed = 113L) {
  dataset <- keras$datasets$boston_housing$load_data(
    path = path,
    seed = as.integer(seed),
    test_split = test_split
  )
  as_dataset_list(dataset)
}




as_dataset_list <- function(dataset) {
  list(
    train = list(
      x = dataset[[1]][[1]],
      y = dataset[[1]][[2]]
    ),
    test = list(
      x = dataset[[2]][[1]],
      y = dataset[[2]][[2]]
    )
  )
}

as_sequences_dataset_list <- function(dataset) {
  list(
    train = list(
      x = lapply(dataset[[1]][[1]], identity),
      y = as.integer(dataset[[1]][[2]])
    ),
    test = list(
      x = lapply(dataset[[2]][[1]], identity),
      y = as.integer(dataset[[2]][[2]])
    )
  )
}
