
context("generators")

source("utils.R")

test_succeeds("image data generator can be used for training", {

  num_classes <- 10
  cifar10 <- dataset_cifar10()
  X_train <- cifar10$train$x
  X_test <- cifar10$test$x
  Y_train <- to_categorical(cifar10$train$y, num_classes)
  Y_test <- to_categorical(cifar10$test$y, num_classes)
  
  X_train <- X_train[1:500,,,]
  X_test <- X_test[1:100,,,]
  Y_train <- Y_train[1:500,]
  Y_test <- Y_test[1:100,]
  
  # create model
  model <- keras_model_sequential()
  model %>% 
    layer_conv_2d(filters = 32, kernel_size = c(3,3), padding = 'same',
                  input_shape = c(32, 32, 3)) %>% 
    layer_activation(activation = 'relu') %>% 
    layer_conv_2d(filters = 32, kernel_size = c(3,3)) %>% 
    layer_activation(activation = 'relu') %>% 
    layer_max_pooling_2d(pool_size = c(2,2)) %>% 
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
  datagen %>% fit_image_data_generator(X_train)
  
  # train using generator
  model %>%
    fit_generator(flow_images_from_data(X_train, Y_train, datagen, batch_size = 32),
                  steps_per_epoch = 32, epochs = 2)
  
  # evaluate using generator
  scores <- model %>%
    evaluate_generator(flow_images_from_data(X_test, Y_test, datagen, batch_size = 32),
                       steps = 5)
  expect_equal(names(scores), c("loss", "acc"))
  
  # predict using generator
  model %>%
    predict_generator(flow_images_from_data(X_test, Y_test, datagen, batch_size = 32),
                      steps = 5)
})

test_succeeds("R function can be used as custom generator", {
 
  # create model
  library(keras)
  model <- keras_model_sequential()
  
  # add layers and compile the model
  model %>% 
    layer_dense(units = 32, activation = 'relu', input_shape = c(100)) %>% 
    layer_dense(units = 1, activation = 'sigmoid') %>% 
    compile(
      optimizer = 'rmsprop',
      loss = 'binary_crossentropy',
      metrics = c('accuracy')
    )
  
  # Generate dummy data
  X_train <- matrix(runif(1000*100), nrow = 1000, ncol = 100)
  Y_train <- matrix(round(runif(1000, min = 0, max = 1)), nrow = 1000, ncol = 1)
  X_test <- matrix(runif(200*100), nrow = 200, ncol = 100)
  Y_test <- matrix(round(runif(200, min = 0, max = 1)), nrow = 200, ncol = 1)
  
  # R generator that draws 32 random elements at a time from the data
  sampling_generator <- function(X_data, Y_data = NULL, batch_size = 32) {
    function() {
      gc() # should blow up R if we are ever called on a background thread
      rows <- sample(1:nrow(X_data), batch_size, replace = TRUE)
      if (!is.null(Y_data))
        list(X_data[rows,], Y_data[rows,])
      else
        list(X_data[rows,])
    }
  }
  
  # Train the model, iterating on the data in batches of 32 samples
  model %>% 
    fit_generator(sampling_generator(X_train, Y_train, batch_size = 32), 
                  steps_per_epoch = 10, epochs = 10)
  
  # Evaluate the model
  model %>% 
    evaluate_generator(sampling_generator(X_test, Y_test, batch_size = 32), 
                       steps = 10)
  
  # generate predictions
  model %>% 
    predict_generator(sampling_generator(X_test, batch_size = 32), 
                      steps = 10)
   
})

