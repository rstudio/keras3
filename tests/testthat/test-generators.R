
context("generators")

source("utils.R")

test_succeeds("image data generator can be used for training", {

  num_classes <- 10
  cifar10 <- dataset_cifar10()
  X_train <- cifar10$train$x
  X_test <- cifar10$test$x
  Y_train <- to_categorical(cifar10$train$y, num_classes)
  Y_test <- to_categorical(cifar10$test$y, num_classes)
  
  # create model
  model <- keras_model_sequential()
  model %>% 
    layer_conv_2d(filters = 32, kernel_size = c(3,3), padding = 'same',
                  input_shape = c(32, 32, 3)) %>% 
    layer_activation(activation = 'relu') %>% 
    layer_conv_2d(filters = 32, kernel_size = c(3,3)) %>% 
    layer_activation(activation = 'relu') %>% 
    layer_max_pooling_2d(pool_size = c(2,2)) %>% 
    layer_dropout(rate = 0.25) %>% 
    layer_flatten() %>% 
    layer_dense(units = num_classes) %>% 
    layer_activation(activation = 'softmax')
  
  # compile model
  model %>% compile(
    loss='categorical_crossentropy',
    optimizer=optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
    metrics=c('accuracy')
  )
  
  # create image data generator
  datagen <- image_data_generator(
    featurewise_center = TRUE,
    featurewise_std_normalization = TRUE,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = TRUE
  )
  datagen %>% fit(X_train)
  
  # train using generator
  model %>%
    fit_generator(image_data_flow(datagen, X_train, Y_train, batch_size = 32),
                  steps_per_epoch = 32, epochs = 2)
 
    
})


