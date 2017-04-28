context("datasets")

source("utils.R")

# these tests take a long time to load so we skip them by default
skip <- is.na(Sys.getenv("KERAS_TEST_DATASETS", unset = NA)) && 
        is.na(Sys.getenv("KERAS_TEST_ALL", unset = NA))

test_dataset <- function(name) {
  if (skip)
    return()
  dataset_fn <- eval(parse(text = paste0("dataset_", name)))
  test_call_succeeds(name, {
    dataset_fn()
  })
}

test_dataset("cifar10")
test_dataset("cifar100")
test_dataset("imdb")
test_dataset("reuters")
test_dataset("reuters_word_index")
test_dataset("mnist")
test_dataset("boston_housing")




