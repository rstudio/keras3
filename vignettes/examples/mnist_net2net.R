# This is an implementation of Net2Net experiment with MNIST in
# Net2Net: Accelerating Learning via Knowledge Transfer'
# by Tianqi Chen, Ian Goodfellow, and Jonathon Shlens
# 
# arXiv:1511.05641v4 [cs.LG] 23 Apr 2016
# http://arxiv.org/abs/1511.05641
# 
# Notes
# - What:
# + Net2Net is a group of methods to transfer knowledge from a teacher neural
# net to a student net,so that the student net can be trained faster than
# from scratch.
# + The paper discussed two specific methods of Net2Net, i.e. Net2WiderNet
# and Net2DeeperNet.
# + Net2WiderNet replaces a model with an equivalent wider model that has
# more units in each hidden layer.
# + Net2DeeperNet replaces a model with an equivalent deeper model.
# + Both are based on the idea of 'function-preserving transformations of
# neural nets'.
# - Why:
# + Enable fast exploration of multiple neural nets in experimentation and
# design process,by creating a series of wider and deeper models with
# transferable knowledge.
# + Enable 'lifelong learning system' by gradually adjusting model complexity
# to data availability,and reusing transferable knowledge.
# 
# Experiments
# - Teacher model: a basic CNN model trained on MNIST for 3 epochs.
# - Net2WiderNet experiment:
# + Student model has a wider Conv2D layer and a wider FC layer.
# + Comparison of 'random-padding' vs 'net2wider' weight initialization.
# + With both methods, student model should immediately perform as well as
# teacher model, but 'net2wider' is slightly better.
# - Net2DeeperNet experiment:
# + Student model has an extra Conv2D layer and an extra FC layer.
# + Comparison of 'random-init' vs 'net2deeper' weight initialization.
# + Starting performance of 'net2deeper' is better than 'random-init'.
# - Hyper-parameters:
# + SGD with momentum=0.9 is used for training teacher and student models.
# + Learning rate adjustment: it's suggested to reduce learning rate
#     to 1/10 for student model.
#   + Addition of noise in 'net2wider' is used to break weight symmetry
#     and thus enable full capacity of student models. It is optional
#     when a Dropout layer is used.
# 
# Results
# - Tested with 'Theano' backend and 'channels_first' image_data_format.
# - Running on GPU GeForce GTX 980M
# - Performance Comparisons - validation loss values during first 3 epochs:
# (1) teacher_model:             0.075    0.041    0.041
# (2) wider_random_pad:          0.036    0.034    0.032
# (3) wider_net2wider:           0.032    0.030    0.030
# (4) deeper_random_init:        0.061    0.043    0.041
# (5) deeper_net2deeper:         0.032    0.031    0.029
library(keras)
library(abind)
library(purrr)

# Function definition -----------------------------------------------------

preprocess_input <- function(x){
  x <- x/255
  dim(x) <- c(dim(x), 1)
  x
}

preprocess_output <- function(y, num_classes = 10){
  to_categorical(y, num_classes = num_classes)
}

bincount <- function(x){
  map_int(1:max(x), ~sum(x == .x))
}

# Parameters --------------------------------------------------------------

input_shape <- c(28, 28, 1) # image shape
num_class <- 10 # num_class

# Data Preparation --------------------------------------------------------

mnist <- dataset_mnist()

# preprocess ionput & output
mnist$train$x <- preprocess_input(mnist$train$x)
mnist$test$x <- preprocess_input(mnist$test$x)
mnist$train$y <- preprocess_output(mnist$train$y)
mnist$test$y <- preprocess_output(mnist$test$y)


# Knowledge transfer algorithms -------------------------------------------


# Get initial weights for a wider conv2d layer with a bigger filters,
# by 'random-padding' or 'net2wider'.
# 
#     # Arguments
#         teacher_w1: `weight` of conv2d layer to become wider,
#           of shape (filters1, num_channel1, kh1, kw1)
#         teacher_b1: `bias` of conv2d layer to become wider,
#           of shape (filters1, )
#         teacher_w2: `weight` of next connected conv2d layer,
#           of shape (filters2, num_channel2, kh2, kw2)
#         new_width: new `filters` for the wider conv2d layer
#         init: initialization algorithm for new weights,
#           either 'random-pad' or 'net2wider'
# 
wider2net_conv2d <- function(teacher_w1, teacher_b1, 
                             teacher_w2, new_width, init){
  
  
  n <- new_width - dim(teacher_w1)[4]
  
  if(init == "random-pad"){
    
    shape_new_w1 <- c(dim(teacher_w1)[-4], n)
    new_w1 <- rnorm(prod(shape_new_w1) , 0, 0.1) %>%
      array(dim = shape_new_w1)
    
    new_b1 <- rep(1, n)*0.1
    
    shape_new_w2 <- c(dim(teacher_w2)[1:2], n, dim(teacher_w2)[4])
    new_w2 <- rnorm(prod(shape_new_w2) , 0, 0.1) %>%
      array(dim = shape_new_w2)
    
  } else if (init == "net2wider"){
    
    index <- sample(1:dim(teacher_w1)[4], size = n, replace = TRUE)
    factors <- bincount(index)[index] + 1
    dim(factors) <- c(1, 1, length(factors), 1)
    
    new_w1 <- teacher_w1[,,,index]
    new_b1 <- teacher_b1[index]
    new_w2 <- apply(teacher_w2[,,index,], c(1,2,4), function(x) x/factors) %>%
      aperm(c(2,3,1,4))
  
  }
  
  
  student_w1 <- abind(teacher_w1, new_w1, along = 4)
  
  if(init == "random-pad"){
    student_w2 <- abind(teacher_w2, new_w2, along = 3)
  } else if (init == "net2wider"){
    
    noise <- rnorm(prod(dim(new_w2)), 0, 5e-2*sd(new_w2)) %>%
      array(dim = dim(new_w2))
    student_w2 <- abind(teacher_w2, new_w2 + noise, axis = 3)
    student_w2[,,index,] <- new_w2
  }
  
  student_b1 <- c(teacher_b1, new_b1) %>% array(., dim = c(length(.)))
  
  list(
    student_w1 = student_w1, 
    student_b1 = student_b1, 
    student_w2 = student_w2
  )
}

# methods to construct teacher_model and student_models
make_teacher_model <- function(train_data, validation_data, epochs=3){
  
  model <- keras_model_sequential()
  
  model %>%
    layer_conv_2d(input_shape = input_shape, 64, list(3, 3), padding = "same", name = "conv1") %>%
    layer_max_pooling_2d(2, name = "pool1") %>%
    layer_conv_2d(64, list(3,3), padding = "same", name = "conv2") %>%
    layer_max_pooling_2d(2, name = "pool2") %>%
    layer_flatten(name = "flatten") %>%
    layer_dense(64, activation = "relu", name = "fc1") %>%
    layer_dense(num_class, activation = "softmax", name = "fc2")
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_sgd(lr = 0.01, momentum = 0.9),
    metrics = "accuracy"
  )
  
  history <- model %>% fit(
    train_data$x, train_data$y,
    epochs = epochs,
    validation_data = list(validation_data$x, validation_data$y)
  )
  
  model
}

make_wider_model_scratch <- function(train_data, validation_data, epochs=3){
  
  new_conv1_width <- 128
  new_fc1_width <- 128
  
  model <- keras_model_sequential()
  
  model %>%
    # a wider conv1 compared to teacher_model
    layer_conv_2d(
      input_shape = input_shape, new_conv1_width, 
      list(3, 3), padding = "same", name = "conv1"
    ) %>%
    layer_max_pooling_2d(2, name = "pool1") %>%
    layer_conv_2d(64, list(3,3), padding = "same", name = "conv2") %>%
    layer_max_pooling_2d(2, name = "pool2") %>%
    layer_flatten(name = "flatten") %>%
    # a wider fc1 compared to teacher model
    layer_dense(new_fc1_width, activation = "relu", name = "fc1") %>%
    layer_dense(num_class, activation = "softmax", name = "fc2")
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_sgd(lr = 0.01, momentum = 0.9),
    metrics = "accuracy"
  )
  
  history <- model %>% fit(
    train_data$x, train_data$y,
    epochs = epochs,
    validation_data = list(validation_data$x, validation_data$y)
  )
  
  model
}



# Train a wider student model based on teacher_model,
# with either 'random-pad' (baseline) or 'net2wider'
make_wider_student_model <- function(teacher_model, train_data,
                                     validation_data, init, epochs=3){
  
  new_conv1_width <- 128
  new_fc1_width <- 128
  
  model <- keras_model_sequential()
  
  model %>%
    # a wider conv1 compared to teacher_model
    layer_conv_2d(
      input_shape = input_shape, new_conv1_width, 
      list(3, 3), padding = "same", name = "conv1"
    ) %>%
    layer_max_pooling_2d(2, name = "pool1") %>%
    layer_conv_2d(64, list(3,3), padding = "same", name = "conv2") %>%
    layer_max_pooling_2d(2, name = "pool2") %>%
    layer_flatten(name = "flatten") %>%
    # a wider fc1 compared to teacher model
    layer_dense(new_fc1_width, activation = "relu", name = "fc1") %>%
    layer_dense(num_class, activation = "softmax", name = "fc2")
  
  # The weights for other layers need to be copied from teacher_model
  # to student_model, except for widened layers
  # and their immediate downstreams, which will be initialized separately.
  # For this example there are no other layers that need to be copied.
  
  weights_conv1 <- teacher_model %>% 
    get_layer(name = "conv1") %>%
    get_weights()
  
  weights_conv2 <- teacher_model %>% 
    get_layer(name = "conv2") %>%
    get_weights()
  
  new_weights <- wider2net_conv2d(weights_conv1[[1]], weights_conv1[[2]], weights_conv2[[1]], 
                        new_conv1_width, "random-pad")
  
  model %>%
    get_layer("conv1") %>%
    set_weights(list(
      new_weights$student_w1,
      new_weights$student_b1
    ))
  
  model %>%
    get_layer("conv2") %>%
    set_weights(list(
     new_weights$student_w2,
     weights_conv2[[2]]
    ))
  
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_sgd(lr = 0.001, momentum = 0.9),
    metrics = "accuracy"
    )
  
  model %>% fit(
    train_data$x, train_data$y,
    epochs = epochs,
    validation_data = list(validation_data$x, validation_data$y)
    )
  
  model
}

# experiments setup

# Benchmark performances of
# (1) a teacher model,
# (2) a wider student model with `random_pad` initializer
# (3) a wider student model with `Net2WiderNet` initializer
net2wider_experiment <- function(){
  
  cat("\nExperiment of Net2WiderNet ...")
  cat("\nbuilding teacher model ...\n")
  
  teacher_model <- make_teacher_model(mnist$train, mnist$test, epochs = 3)

  cat("\nbuilding wider student model by random padding ...")
  make_wider_student_model(teacher_model, mnist$train,
                           mnist$test, "random-pad",
                           epochs=3)
  
  
  cat('\nbuilding wider student model by net2wider ...')
  make_wider_student_model(teacher_model, mnist$train,
                           mnist$test, "net2wider",
                           epochs=3)
  
  cat("\nbuilding widedr model from scratch")
  make_wider_model_scratch(mnist$train, mnist$test, epochs = 3)
  
  cat("\nFinished")
}


net2wider_experiment()
 