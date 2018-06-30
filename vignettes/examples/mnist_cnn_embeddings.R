#' This example shows how to visualize embeddings in TensorBoard.
#' 
#' Embeddings in the sense used here don't necessarily refer to embedding layers.
#' In fact, features (= activations) from other hidden layers can be visualized,
#' as shown in this example for a dense layer.

library(keras)

# Data Preparation -----------------------------------------------------

batch_size <- 128
num_classes <- 10
epochs <- 12

# Input image dimensions
img_rows <- 28
img_cols <- 28

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Redefine  dimension of train/test inputs
x_train <-
  array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <-
  array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')


# Prepare for logging embeddings --------------------------------------------------

embeddings_dir <- file.path(tempdir(), 'embeddings')
if (!file.exists(embeddings_dir))
  dir.create(embeddings_dir)
embeddings_metadata <- file.path(embeddings_dir, 'metadata.tsv')

# we use the class names from the test set as embeddings_metadata
readr::write_tsv(data.frame(y_test), path = embeddings_metadata, col_names = FALSE)

tensorboard_callback <- callback_tensorboard(
  log_dir = embeddings_dir,
  batch_size = batch_size,
  embeddings_freq = 1,
  # if missing or NULL all embedding layers will be monitored
  embeddings_layer_names = list('features'),
  # single file for all embedding layers, could also be a named list mapping
  # layer names to file names
  embeddings_metadata = embeddings_metadata,
  # data to be embedded
  embeddings_data = x_test
)


# Define Model -----------------------------------------------------------

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# Define model
model <- keras_model_sequential() %>%
  layer_conv_2d(
    filters = 32,
    kernel_size = c(3, 3),
    activation = 'relu',
    input_shape = input_shape
  ) %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  # these are the embeddings (activations) we are going to visualize
  layer_dense(units = 128, activation = 'relu', name = 'features') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = num_classes, activation = 'softmax')

# Compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Launch TensorBoard
#
# As the model is being fit you will be able to view the embedings in the 
# Projector tab. On the left, use "color by label" to see the digits displayed
# in 10 different colors. Hover over a point to see its label.
tensorboard(embeddings_dir)

# Train model
model %>% fit(
  x_train,
  y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_data = list(x_test, y_test),
  callbacks = list(tensorboard_callback)
)

scores <- model %>% evaluate(x_test, y_test, verbose = 0)

# Output metrics
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')


