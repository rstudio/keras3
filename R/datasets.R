

#' Loads the MNIST dataset.
#' 
#' @param path path where to cache the dataset locally (relative to ~/.keras/datasets).
#' 
#' @return Lists of training and test data `train$x, train$y, test$x, test$y`.
#' 
#' @export
dataset_mnist <- function(path = "mnist.npz") {
  dataset <- keras$datasets$mnist$load_data(path)
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