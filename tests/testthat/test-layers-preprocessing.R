context("layers-preprocessing")

dataset_mnist_mini <- local({
  mnist_mini <- NULL
  function() {
    if (is.null(mnist_mini)) {
      mnist <- dataset_mnist()
      mnist_mini <- list(x = mnist$test$x[1:50, ,] / 255,
                         y = mnist$test$y[1:50])
      dim(mnist_mini$x) <- c(dim(mnist_mini$x), 1)
      mnist_mini <<- mnist_mini
    }
    mnist_mini
  }
})


peek_py_iterator <- function(x) {
  reticulate::iter_next(reticulate::as_iterator(x))
}

test_image_preprocessing_layer <- function(lyr, ...) {
  if(is_mac_arm64()) local_tf_device("CPU")
  # workaround for bug on M1 Macs  until this error is resolved:
  # No registered 'RngReadAndSkip' OpKernel for 'GPU' devices compatible with node {{node RngReadAndSkip}}
  # .  Registered:  device='XLA_CPU_JIT'
  # device='CPU'
  # [Op:RngReadAndSkip]

  test_succeeds(deparse(substitute(lyr)), {

    mnist_mini <- dataset_mnist_mini()

    # in a sequential model
    model <- keras_model_sequential(input_shape = shape(28, 28, 1)) %>%
      lyr(...)
    expect_tensor(model(mnist_mini$x))

    # in a functional model
    lyr_inst <- lyr(...)
    input <- layer_input( shape(28, 28, 1))
    output <- lyr_inst(input)
    model <- keras_model(input, output)
    expect_tensor(model(mnist_mini$x))


    # in a dataset
    mnist_mini_ds <- tfdatasets::tensor_slices_dataset(
      list(tensorflow::as_tensor(mnist_mini$x, "float32"),
           mnist_mini$y))

    layer <- lyr(...)
    ds <- mnist_mini_ds %>%
      tfdatasets::dataset_map(function(x, y) list(layer(x), y))

    expect_tensor(iter_next(as_iterator(ds))[[1]])
  })

}


if(tf_version() >= "2.9") {
  test_image_preprocessing_layer(layer_random_brightness, factor = .2)
}


if (tf_version() >= "2.6") {

# image preprocessing
test_image_preprocessing_layer(layer_resizing, height = 20, width = 20)
test_image_preprocessing_layer(layer_rescaling, scale = 1/255)
test_image_preprocessing_layer(layer_center_crop, height = 20, width = 20)

# image augmentation
# lyr <- layer_random_crop
test_image_preprocessing_layer(layer_random_crop, height = 20, width = 20)

test_image_preprocessing_layer(layer_random_flip)
test_image_preprocessing_layer(layer_random_translation, height_factor = .5, width_factor = .5)
test_image_preprocessing_layer(layer_random_rotation, factor = 2)
test_image_preprocessing_layer(layer_random_zoom, height_factor = .5)
test_image_preprocessing_layer(layer_random_contrast, factor = .5)
test_image_preprocessing_layer(layer_random_height, factor = .5)
test_image_preprocessing_layer(layer_random_width, factor = .5)
}



if (tf_version() >= "2.6")
test_succeeds("layer_category_encoding", {

  layer <- layer_category_encoding(num_tokens=4, output_mode="one_hot")
  inp <- as.integer(c(3, 2, 0, 1))
  out <- layer(inp)
  expect_tensor(out, shape = c(4L, 4L))

  layer <- layer_category_encoding(num_tokens=4, output_mode="multi_hot")
  inp <- rbind(c(0, 1), c(0, 0), c(1, 2), c(3, 1)) %>% as_tensor("int32")
  out <- layer(inp)
  expect_tensor(out, shape = c(4L, 4L))

  layer <- layer_category_encoding(num_tokens=4, output_mode="count")
  inp <- rbind(c(0, 1), c(0, 0), c(1, 2), c(3, 1)) %>% as_tensor("int32")
  count_weights <- rbind(c(.1, .2), c(.1, .1), c(.2, .3), c(.4, .2))
  out <- layer(inp, count_weights = count_weights)
  expect_tensor(out, shape = c(4L, 4L))

})


if (tf_version() >= "2.6")
test_succeeds("layer_hashing", {
  # **Example (FarmHash64)**
  layer <- layer_hashing(num_bins = 3)
  inp <- matrix(c('A', 'B', 'C', 'D', 'E'))
  expect_tensor(layer(inp), shape = c(5L, 1L))

  # **Example (FarmHash64) with a mask value**
  layer <- layer_hashing(num_bins = 3, mask_value = '')
  inp <- matrix(c('A', 'B', 'C', 'D', 'E'))
  expect_tensor(layer(inp), shape = c(5L, 1L))

  # **Example (SipHash64)**
  layer <- layer_hashing(num_bins = 3, salt = c(133, 137))
  inp <- matrix(c('A', 'B', 'C', 'D', 'E'))
  expect_tensor(layer(inp), shape = c(5L, 1L))


  # **Example (Siphash64 with a single integer, same as `salt=[133, 133]`)**
  layer <- layer_hashing(num_bins = 3, salt = 133)
  inp <- matrix(c('A', 'B', 'C', 'D', 'E'))
  expect_tensor(layer(inp), shape = c(5L, 1L))

})


if (tf_version() >= "2.6")
test_succeeds("layer_integer_lookup", {

  #Creating a lookup layer with a known vocabulary
  vocab = as.integer(c(12, 36, 1138, 42))
  data = as_tensor(rbind(c(12, 1138, 42),
                         c(42, 1000, 36)), "int32")  # Note OOV tokens
  layer = layer_integer_lookup(vocabulary = vocab)
  expect_tensor(layer(data), shape = c(2L, 3L))

  # Creating a lookup layer with an adapted vocabulary
  data = as_tensor(rbind(c(12, 1138, 42),
                         c(42, 1000, 36)), "int32")
  layer = layer_integer_lookup()
  vocab <- layer %>% adapt(data) %>% get_vocabulary()
  expect_equal(vocab, list(-1L, 42L, 1138L, 1000L, 36L, 12L))
  out <- layer(data)
  expect_tensor(out, shape = c(2L, 3L))
  expect_equal(as.array(out), rbind(c(5, 2, 1),
                                    c(1, 3, 4)))

  # Lookups with multiple OOV indices
  vocab = as.integer(c(12, 36, 1138, 42))
  data = as_tensor(rbind(c(12, 1138, 42),
                         c(37, 1000, 36)), "int32")
  layer = layer_integer_lookup(vocabulary=vocab, num_oov_indices=2)
  layer(data)
  expect_tensor(layer(data), shape = c(2L, 3L))

  # One-hot output
  vocab = as.integer(c(12, 36, 1138, 42))
  data = as.integer(c(12, 36, 1138, 42, 7)) # Note OOV tokens
  layer = layer_integer_lookup(vocabulary = vocab, output_mode = 'one_hot')
  expect_tensor(layer(data), shape = c(5L, 5L))

})


if (tf_version() >= "2.6")
test_succeeds("layer_string_lookup", {


  #Creating a lookup layer with a known vocabulary
  vocab = c("a", "b", "c", "d")
  data = as_tensor(rbind(c("a", "c", "d"),
                         c("d", "z", "b")))  # Note OOV tokens
  layer = layer_string_lookup(vocabulary = vocab)
  expect_tensor(layer(data), shape = c(2L, 3L))

  # Creating a lookup layer with an adapted vocabulary
  data = as_tensor(rbind(c("a", "c", "d"),
                         c("d", "z", "b")))  # Note OOV tokens
  layer = layer_string_lookup()
  vocab <- layer %>% adapt(data) %>% get_vocabulary()
  expect_equal(vocab, c("[UNK]", "d", "z", "c", "b", "a"))
  out <- layer(data)
  expect_tensor(out, shape = c(2L, 3L))
  expect_equal(as.array(out), rbind(c(5, 3, 1),
                                    c(1, 2, 4)))

})


if (tf_version() >= "2.6")
test_succeeds("layer_normalization", {

  #Calculate a global mean and variance by analyzing the dataset in adapt().
adapt_data = c(1, 2, 3, 4, 5)
input_data = c(1, 2, 3)
layer = layer_normalization(axis=NULL)
layer %>% adapt(adapt_data)
out <- layer(input_data)
expect_tensor(out, shape = c(3L))
expect_equal(as.numeric(out), c(-1.41421353816986, -0.70710676908493, 0),
             tolerance = 1e-7)

# Calculate a mean and variance for each index on the last axis.
adapt_data = rbind(c(0, 7, 4),
                   c(2, 9, 6),
                   c(0, 7, 4),
                   c(2, 9, 6))
input_data = adapt_data[1:2,]
input_data = rbind(c(0, 7, 4))
layer = layer_normalization(axis=-1)
layer %>% adapt(as_tensor(adapt_data))
out <- layer(input_data)
expect_tensor(out, shape = c(1L, 3L))
out <- as.array(out)


mean     <- as.array(layer$mean)
var <- as.array(layer$variance)
expect_equal(mean, rbind(c(1,8,5)))
expect_equal(var, rbind(c(1,1,1)))
out_manual <- as.array(input_data - mean / sqrt(var))

expect_equal(out, out_manual)
expect_equal(as.vector(out), c(-1, -1, -1))

# Pass the mean and variance directly.
input_data = as_tensor(rbind(1, 2, 3), "float32")
layer = layer_normalization(mean=3, variance=2)
out <- layer(input_data)
expect_tensor(out, shape = c(3L, 1L))
expect_equal(as.array(out), rbind(-1.41421353816986, -0.70710676908493, 0),
             tolerance = 1e-7)


# adapt multiple times in a model
layer = layer_normalization(axis=NULL)
layer %>% adapt(c(0, 2))
model = keras_model_sequential(layer)
out <- model %>% predict(c(0, 1, 2))
expect_equal(out, array(c(-1, 0, 1)))

layer %>% adapt(c(-1, 1))
model %>% compile() # This is needed to re-compile model.predict!
out <- model %>% predict(c(0, 1, 2))
expect_equal(out, array(c(0, 1, 2)))


# adapt multiple times in a dataset
layer = layer_normalization(axis=NULL)
layer %>% adapt(c(0, 2))
input_ds = tfdatasets::range_dataset(0, 3)
normalized_ds = tfdatasets::dataset_map(input_ds, layer)
out <- iterate(normalized_ds$as_numpy_iterator(), simplify = FALSE)
expect_equal(out, list(array(-1), array(0), array(1)))

layer %>% adapt(c(-1, 1))
normalized_ds = tfdatasets::dataset_map(input_ds, layer) # Re-map over the input dataset.
out <- iterate(normalized_ds$as_numpy_iterator(), simplify = FALSE)
expect_equal(out, list(array(0), array(1), array(2)))

})

if (tf_version() >= "2.6")
test_succeeds("layer_discretization", {
  input = rbind(c(-1.5, 1.0, 3.4, .5),
                c(0.0, 3.0, 1.3, 0.0))
  layer = layer_discretization(bin_boundaries = c(0, 1, 2))
  out <- layer(input)
  expect_tensor(out, shape = c(2L, 4L))
  expect_true(out$dtype$is_integer)
  expect_equal(as.array(out), rbind(c(0, 2, 3, 1),
                                    c(1, 3, 2, 1)))

  layer = layer_discretization(num_bins = 4, epsilon = 0.01)
  layer %>% adapt(input)
  out <- layer(input)
  expect_tensor(out, shape = c(2L, 4L))
  expect_true(out$dtype$is_integer)
  expect_equal(as.array(out), rbind(c(0, 2, 3, 2),
                                    c(1, 3, 3, 1)))
})


test_succeeds("layer_text_vectorization", {

  text_dataset = tfdatasets::tensor_slices_dataset(c("foo", "bar", "baz"))

  vectorize_layer = layer_text_vectorization(
    max_tokens = 5000,
    output_mode = 'int',
    output_sequence_length = 4
  )
  vectorize_layer %>%
    adapt(tfdatasets::dataset_batch(text_dataset, 64))


  model <- keras_model_sequential(
    layers = vectorize_layer,
    input_shape = c(1), dtype = tf$string)

  input_data = rbind("foo qux bar", "qux baz")
  preds <- model %>% predict(input_data)

  expect_equal(preds, rbind(c(2, 1, 4, 0),
                            c(1, 3, 0, 0)))


  vocab_data = c("earth", "wind", "and", "fire")
  max_len = 4  # Sequence length to pad the outputs to.


  if(tf_version() >= "2.4") {
  # setting vocab on instantiation not supported prior to 2.4, missing kwarg 'vocabulary'
  vectorize_layer = layer_text_vectorization(
    max_tokens = 5000,
    output_mode = 'int',
    output_sequence_length = 4,
    vocabulary = vocab_data
  )

  vocab <- get_vocabulary(vectorize_layer)
  expect_equal(vocab, c("", "[UNK]", "earth", "wind", "and", "fire"))
  }

})
