# Train a simple deep CNN on the CIFAR10 small images dataset.
#  
# It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
# (it's still underfitting at that point, though).

library(keras)


# Parameters --------------------------------------------------------------

batch_size <- 32
epochs <- 200
data_augmentation <- TRUE


# Data Preparation --------------------------------------------------------

# see ?dataset_cifar10 for more info
cifar10 <- dataset_cifar10()

x_train <- cifar10$train$x/255
x_test <- cifar10$test$x/255

y_train <- cifar10$train$y %>%
  to_categorical(num_classes = 10)

y_test <- cifar10$test$y %>%
  to_categorical(num_classes = 10)

# Defining the model ------------------------------------------------------

model <- keras_model_sequential()

model %>%
  layer_conv_2d(
    filter = 32, kernel_size = c(3,3), padding = "same", 
    input_shape = c(32, 32, 3)
  ) %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(10) %>%
  layer_activation("softmax")

opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)


# Training ----------------------------------------------------------------

if(!data_augmentation){
  
  model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = list(x_test, y_test),
    shuffle = TRUE
  )
  
} else {
  
  datagen <- image_data_generator(
    featurewise_center = TRUE,
    featurewise_std_normalization = TRUE,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = TRUE
  )
  
  datagen %>% fit(x_train)
  
  model %>% fit_generator(
    image_data_flow(datagen, x_train, y_train, batch_size = batch_size),
    steps_per_epoch = as.integer(50000/batch_size), 
    epochs = epochs, 
    validation_data = list(x_test, y_test)
  )
  
}