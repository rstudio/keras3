#' MNIST dataset with TFRecords, the standard TensorFlow data format.
#'
#' TFRecord is a data format supported throughout TensorFlow. This example
#' demonstrates how to load TFRecord data using Input Tensors. Input Tensors
#' differ from the normal Keras workflow because instead of fitting to data
#' loaded into a a numpy array, data is supplied via a special tensor that reads
#' data from nodes that are wired directly into model graph with the
#' `layer_input(tensor=input_tensor)` parameter.
#'
#' There are several advantages to using Input Tensors. First, if a dataset is
#' already in TFRecord format you can load and train on that data directly in
#' Keras. Second, extended backend API capabilities such as TensorFlow data
#' augmentation is easy to integrate directly into your Keras training scripts
#' via input tensors. Third, TensorFlow implements several data APIs for
#' TFRecords, some of which provide significantly faster training performance
#' than numpy arrays can provide because they run via the C++ backend. Please
#' note that this example is tailored for brevity and clarity and not to
#' demonstrate performance or augmentation capabilities.
#'
#' Input Tensors also have important disadvantages. In particular, Input Tensors
#' are fixed at model construction because rewiring networks is not yet
#' supported. For this reason, changing the data input source means model
#' weights must be saved and the model rebuilt from scratch to connect the new
#' input data. validation cannot currently be performed as training progresses,
#' and must be performed after training completes. This example demonstrates how
#' to train with input tensors, save the model weights, and then evaluate the
#' model using the standard Keras API.
#'
#' Gets to ~99.1% validation accuracy after 5 epochs (there is still a lot of margin
#' for parameter tuning).
#' 
#' 

library(keras)
library(tensorflow)

if (k_backend() != 'tensorflow') {
  stop('This example can only run with the ',
       'TensorFlow backend, ',
       'because it requires TFRecords, which ',
       'are not supported on other platforms.')
}

# Define Model -------------------------------------------------------------------

cnn_layers <- function(x_train_input) {
  x_train_input %>% 
    layer_conv_2d(filters = 32, kernel_size = c(3,3), 
                  activation = 'relu', padding = 'valid') %>% 
    layer_max_pooling_2d(pool_size = c(2,2)) %>% 
    layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
    layer_max_pooling_2d(pool_size = c(2,2)) %>% 
    layer_flatten() %>% 
    layer_dense(units = 512, activation = 'relu') %>% 
    layer_dropout(rate = 0.5) %>% 
    layer_dense(units = classes, activation = 'softmax', name = 'x_train_out')
}

sess <- k_get_session()

# Data Preparation --------------------------------------------------------------

batch_size <- 128L
batch_shape = list(batch_size, 28L, 28L, 1L)
steps_per_epoch <- 469L
epochs <- 5L
classes <- 10L

# The capacity variable controls the maximum queue size
# allowed when prefetching data for training.
capacity <- 10000L

# min_after_dequeue is the minimum number elements in the queue
# after a dequeue, which ensures sufficient mixing of elements.
min_after_dequeue <- 3000L

# If `enqueue_many` is `FALSE`, `tensors` is assumed to represent a
# single example.  An input tensor with shape `(x, y, z)` will be output
# as a tensor with shape `(batch_size, x, y, z)`.
#
# If `enqueue_many` is `TRUE`, `tensors` is assumed to represent a
# batch of examples, where the first dimension is indexed by example,
# and all members of `tensors` should have the same size in the
# first dimension.  If an input tensor has shape `(*, x, y, z)`, the
# output will have shape `(batch_size, x, y, z)`.
enqueue_many <- TRUE

# mnist dataset from tf contrib
mnist <- tf$contrib$learn$datasets$mnist
data <- mnist$load_mnist()

train_data <- tf$train$shuffle_batch(
  tensors = list(data$train$images, data$train$labels),
  batch_size = batch_size,
  capacity = capacity,
  min_after_dequeue = min_after_dequeue,
  enqueue_many = enqueue_many,
  num_threads = 8L
)
x_train_batch <- train_data[[1]]
y_train_batch <- train_data[[2]]

x_train_batch <- tf$cast(x_train_batch, tf$float32)
x_train_batch <- tf$reshape(x_train_batch, shape = batch_shape)

y_train_batch <- tf$cast(y_train_batch, tf$int32)
y_train_batch <- tf$one_hot(y_train_batch, classes)

x_batch_shape <- x_train_batch$get_shape()$as_list()
y_batch_shape = y_train_batch$get_shape()$as_list()

x_train_input <- layer_input(tensor = x_train_batch, batch_shape = x_batch_shape)
x_train_out <- cnn_layers(x_train_input)

# Training & Evaluation ---------------------------------------------------------

train_model = keras_model(inputs = x_train_input, outputs = x_train_out)

# Pass the target tensor `y_train_batch` to `compile`
# via the `target_tensors` keyword argument:
train_model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-3, decay = 1e-5),
  loss = 'categorical_crossentropy',
  metrics = c('accuracy'),
  target_tensors = y_train_batch
)

summary(train_model)

# Fit the model using data from the TFRecord data tensors.
coord <- tf$train$Coordinator()
threads = tf$train$start_queue_runners(sess, coord)

train_model %>% fit(
  epochs = epochs,
  steps_per_epoch = steps_per_epoch
)

# Save the model weights.
train_model %>% save_model_weights_hdf5('saved_wt.h5')

# Clean up the TF session.
coord$request_stop()
coord$join(threads)
k_clear_session()

# Second Session to test loading trained model without tensors
x_test <- data$validation$images
x_test <- array_reshape(x_test, dim = c(nrow(x_test), 28, 28, 1))
y_test <- data$validation$labels
x_test_inp <- layer_input(shape = dim(x_test)[-1])
test_out <- cnn_layers(x_test_inp)
test_model <- keras_model(inputs = x_test_inp, outputs = test_out)
test_model %>% load_model_weights_hdf5('saved_wt.h5')
test_model %>% compile(
  optimizer = 'rmsprop', 
  loss = 'categorical_crossentropy', 
  metrics = c('accuracy')
)
summary(test_model)

result <- test_model %>% evaluate(x_test, to_categorical(y_test, classes))
cat(sprintf('\nTest accuracy: %f', result$acc))
