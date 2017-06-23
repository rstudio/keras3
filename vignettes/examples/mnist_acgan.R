#' Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
#' MNIST dataset. See https://arxiv.org/abs/1610.09585 for more details.
#'
#' You should start to see reasonable images after ~5 epochs, and good images by
#' ~15 epochs. You should use a GPU, as the convolution-heavy operations are
#' very slow on the CPU. Prefer the TensorFlow backend if you plan on iterating,
#' as the compilation time can be a blocker using Theano.
#'
#' Timings:
#'   
#'   Hardware           | Backend | Time / Epoch
#' -------------------------------------------
#'   CPU               | TF       | 3 hrs  
#'   Titan X (maxwell) | TF       | 4 min
#'   Titan X (maxwell) | TH       | 7 min
#' 

library(keras)
library(progress)
library(abind)
K <- keras::backend()
K$set_image_data_format('channels_first')

# Functions ---------------------------------------------------------------

build_generator <- function(latent_size){
  
  # we will map a pair of (z, L), where z is a latent vector and L is a
  # label drawn from P_c, to image space (..., 1, 28, 28)
  cnn <- keras_model_sequential()
  
  cnn %>%
    layer_dense(1024, input_shape = latent_size, activation = "relu") %>%
    layer_dense(128*7*7, activation = "relu") %>%
    layer_reshape(c(128, 7, 7)) %>%
    # upsample to (..., 14, 14)
    layer_upsampling_2d(size = c(2, 2)) %>%
    layer_conv_2d(
      256, c(5,5), padding = "same", activation = "relu",
      kernel_initializer = "glorot_normal"
    ) %>%
    # upsample to (..., 28, 28)
    layer_upsampling_2d(size = c(2, 2)) %>%
    layer_conv_2d(
      128, c(5,5), padding = "same", activation = "tanh",
      kernel_initializer = "glorot_normal"
    ) %>%
    # take a channel axis reduction
    layer_conv_2d(
      1, c(2,2), padding = "same", activation = "tanh",
      kernel_initializer = "glorot_normal"
    )
  
  
  # this is the z space commonly refered to in GAN papers
  latent <- layer_input(shape = list(latent_size))
  
  # this will be our label
  image_class <- layer_input(shape = list(1))
  
  # 10 classes in MNIST
  cls <-  image_class %>%
    layer_embedding(
      input_dim = 10, output_dim = latent_size, 
      embeddings_initializer='glorot_normal'
    ) %>%
    layer_flatten()
  
  
  # hadamard product between z-space and a class conditional embedding
  h <- layer_multiply(list(latent, cls))
  
  fake_image <- cnn(h)
  
  keras_model(list(latent, image_class), fake_image)
}

build_discriminator <- function(){
  
  # build a relatively standard conv net, with LeakyReLUs as suggested in
  # the reference paper
  cnn <- keras_model_sequential()
  
  cnn %>%
    layer_conv_2d(
      32, c(3,3), padding = "same", strides = c(2,2),
      input_shape = c(1, 28, 28)
    ) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%
    
    layer_conv_2d(64, c(3, 3), padding = "same", strides = c(1,1)) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%  
    
    layer_conv_2d(128, c(3, 3), padding = "same", strides = c(2,2)) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%  
    
    layer_conv_2d(256, c(3, 3), padding = "same", strides = c(1,1)) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%  
    
    layer_flatten()
  
  
  
  image <- layer_input(shape = c(1, 28, 28))
  features <- cnn(image)
  
  # first output (name=generation) is whether or not the discriminator
  # thinks the image that is being shown is fake, and the second output
  # (name=auxiliary) is the class that the discriminator thinks the image
  # belongs to.
  fake <- features %>% 
    layer_dense(1, activation = "sigmoid", name = "generation")
  
  aux <- features %>%
    layer_dense(10, activation = "softmax", name = "auxiliary")
  
  keras_model(image, list(fake, aux))
}

# Parameters --------------------------------------------------------------

# batch and latent size taken from the paper
epochs <- 50
batch_size <- 100
latent_size <- 100

# Adam parameters suggested in https://arxiv.org/abs/1511.06434
adam_lr <- 0.00005 
adam_beta_1 <- 0.5

# Model definition --------------------------------------------------------

# build the discriminator
discriminator <- build_discriminator()
discriminator %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
  loss = list("binary_crossentropy", "sparse_categorical_crossentropy")
)

# build the generator
generator <- build_generator(latent_size)
generator %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
  loss = "binary_crossentropy"
)

latent <- layer_input(shape = list(latent_size))
image_class <- layer_input(shape = list(1), dtype = "int32")

fake <- generator(list(latent, image_class))

# we only want to be able to train generation for the combined model

discriminator$trainable <- FALSE
results <- discriminator(fake)

combined <- keras_model(list(latent, image_class), results)
combined %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
  loss = list("binary_crossentropy", "sparse_categorical_crossentropy")
)


# Data preparation --------------------------------------------------------

# get our mnist data, and force it to be of shape (..., 1, 28, 28) with
# range [-1, 1]
mnist <- dataset_mnist()
mnist$train$x <- (mnist$train$x - 127.5)/127.5
mnist$test$x <- (mnist$test$x - 127.5)/127.5
dim(mnist$train$x) <- c(60000, 1, 28, 28) 
dim(mnist$test$x) <- c(10000, 1, 28, 28) 

num_train <- dim(mnist$train$x)[1]
num_test <- dim(mnist$test$x)[1]

# Training ----------------------------------------------------------------

for(epoch in 1:epochs){
  
  num_batches <- trunc(num_train/batch_size)
  pb <- progress_bar$new(
    total = num_batches, 
    format = sprintf("epoch %s/%s :elapsed [:bar] :percent :eta", epoch, epochs),
    clear = FALSE
  )
  
  epoch_gen_loss <- NULL
  epoch_disc_loss <- NULL
  
  possible_indexes <- 1:num_train
  
  for(index in 1:num_batches){
    
    pb$tick()
    
    # generate a new batch of noise
    noise <- runif(n = batch_size*latent_size, min = -1, max = 1) %>%
      matrix(nrow = batch_size, ncol = latent_size)
    
    # get a batch of real images
    batch <- sample(possible_indexes, size = batch_size)
    possible_indexes <- possible_indexes[!possible_indexes %in% batch]
    image_batch <- mnist$train$x[batch,,,,drop = FALSE]
    label_batch <- mnist$train$y[batch]
    
    # sample some labels from p_c
    sampled_labels <- sample(0:9, batch_size, replace = TRUE) %>%
      matrix(ncol = 1)
    
    # generate a batch of fake images, using the generated labels as a
    # conditioner. We reshape the sampled labels to be
    # (batch_size, 1) so that we can feed them into the embedding
    # layer as a length one sequence
    generated_images <- predict(generator, list(noise, sampled_labels))
    
    X <- abind(image_batch, generated_images, along = 1)
    y <- c(rep(1L, batch_size), rep(0L, batch_size)) %>% matrix(ncol = 1)
    aux_y <- c(label_batch, sampled_labels) %>% matrix(ncol = 1)
    
    # see if the discriminator can figure itself out...
    disc_loss <- train_on_batch(
      discriminator, x = X, 
      y = list(y, aux_y)
    )
    
    epoch_disc_loss <- rbind(epoch_disc_loss, unlist(disc_loss))
    
    # make new noise. we generate 2 * batch size here such that we have
    # the generator optimize over an identical number of images as the
    # discriminator
    noise <- runif(2*batch_size*latent_size, min = -1, max = 1) %>%
      matrix(nrow = 2*batch_size, ncol = latent_size)
    sampled_labels <- sample(0:9, size = 2*batch_size, replace = TRUE) %>%
      matrix(ncol = 1)
    
    # we want to train the generator to trick the discriminator
    # For the generator, we want all the {fake, not-fake} labels to say
    # not-fake
    trick <- rep(1, 2*batch_size) %>% matrix(ncol = 1)
    
    combined_loss <- train_on_batch(
      combined, 
      list(noise, sampled_labels),
      list(trick, sampled_labels)
    )
    
    epoch_gen_loss <- rbind(epoch_gen_loss, unlist(combined_loss))
    
  }
  
  cat(sprintf("\nTesting for epoch %02d:", epoch))
  
  # evaluate the testing loss here
  
  # generate a new batch of noise
  noise <- runif(num_test*latent_size, min = -1, max = 1) %>%
    matrix(nrow = num_test, ncol = latent_size)
  
  # sample some labels from p_c and generate images from them
  sampled_labels <- sample(0:9, size = num_test, replace = TRUE) %>%
    matrix(ncol = 1)
  generated_images <- predict(generator, list(noise, sampled_labels))
  
  X <- abind(mnist$test$x, generated_images, along = 1)
  y <- c(rep(1, num_test), rep(0, num_test)) %>% matrix(ncol = 1)
  aux_y <- c(mnist$test$y, sampled_labels) %>% matrix(ncol = 1)
  
  # see if the discriminator can figure itself out...
  discriminator_test_loss <- evaluate(
    discriminator, X, list(y, aux_y), 
    verbose = FALSE
  ) %>% unlist()
  
  discriminator_train_loss <- apply(epoch_disc_loss, 2, mean)
  
  # make new noise
  noise <- runif(2*num_test*latent_size, min = -1, max = 1) %>%
    matrix(nrow = 2*num_test, ncol = latent_size)
  sampled_labels <- sample(0:9, size = 2*num_test, replace = TRUE) %>%
    matrix(ncol = 1)
  
  trick <- rep(1, 2*num_test) %>% matrix(ncol = 1)
  
  generator_test_loss = combined %>% evaluate(
    list(noise, sampled_labels),
    list(trick, sampled_labels),
    verbose = FALSE
  )
  
  generator_train_loss <- apply(epoch_gen_loss, 2, mean)
  
  
  # generate an epoch report on performance
  row_fmt <- "\n%22s : loss %4.2f | %5.2f | %5.2f"
  cat(sprintf(
    row_fmt, 
    "generator (train)",
    generator_train_loss[1],
    generator_train_loss[2],
    generator_train_loss[3]
  ))
  cat(sprintf(
    row_fmt, 
    "generator (test)",
    generator_test_loss[1],
    generator_test_loss[2],
    generator_test_loss[3]
  ))
  
  cat(sprintf(
    row_fmt, 
    "discriminator (train)",
    discriminator_train_loss[1],
    discriminator_train_loss[2],
    discriminator_train_loss[3]
  ))
  cat(sprintf(
    row_fmt, 
    "discriminator (test)",
    discriminator_test_loss[1],
    discriminator_test_loss[2],
    discriminator_test_loss[3]
  ))
  
  cat("\n")
  
  # generate some digits to display
  noise <- runif(10*latent_size, min = -1, max = 1) %>%
    matrix(nrow = 10, ncol = latent_size)
  
  sampled_labels <- 0:9 %>%
    matrix(ncol = 1)
  
  # get a batch to display
  generated_images <- predict(
    generator,    
    list(noise, sampled_labels)
  )
  
  img <- NULL
  for(i in 1:10){
    img <- cbind(img, generated_images[i,,,])
  }
  
  ((img + 1)/2) %>% as.raster() %>%
    plot()
  
}


   