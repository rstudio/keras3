#' This is part of the companion code to the post
#' "Representation learning with MMD-VAE"
#' on the TensorFlow for R blog.
#'
#' https://blogs.rstudio.com/tensorflow/posts/2018-10-22-mmd-vae/


library(keras)
use_implementation("tensorflow")
library(tensorflow)
tfe_enable_eager_execution(device_policy = "silent")

library(tfdatasets)
library(dplyr)
library(ggplot2)
library(glue)


# Setup and preprocessing -------------------------------------------------

fashion <- dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashion$train
c(test_images, test_labels) %<-% fashion$test

train_x <-
  train_images %>% `/`(255) %>% k_reshape(c(60000, 28, 28, 1))
test_x <-
  test_images %>% `/`(255) %>% k_reshape(c(10000, 28, 28, 1))

class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

buffer_size <- 60000
batch_size <- 100
batches_per_epoch <- buffer_size / batch_size

train_dataset <- tensor_slices_dataset(train_x) %>%
  dataset_shuffle(buffer_size) %>%
  dataset_batch(batch_size)

test_dataset <- tensor_slices_dataset(test_x) %>%
  dataset_batch(10000)



# Model -------------------------------------------------------------------

latent_dim <- 2

encoder_model <- function(name = NULL) {
  keras_model_custom(name = name, function(self) {
    self$conv1 <-
      layer_conv_2d(
        filters = 32,
        kernel_size = 3,
        strides = 2,
        activation = "relu"
      )
    self$conv2 <-
      layer_conv_2d(
        filters = 64,
        kernel_size = 3,
        strides = 2,
        activation = "relu"
      )
    self$flatten <- layer_flatten()
    self$dense <- layer_dense(units = 2 * latent_dim)
    
    function (x, mask = NULL) {
      x %>%
        self$conv1() %>%
        self$conv2() %>%
        self$flatten() %>%
        self$dense() %>%
        tf$split(num_or_size_splits = 2L, axis = 1L) 
    }
  })
}

decoder_model <- function(name = NULL) {
  keras_model_custom(name = name, function(self) {
    self$dense <- layer_dense(units = 7 * 7 * 32, activation = "relu")
    self$reshape <- layer_reshape(target_shape = c(7, 7, 32))
    self$deconv1 <-
      layer_conv_2d_transpose(
        filters = 64,
        kernel_size = 3,
        strides = 2,
        padding = "same",
        activation = "relu"
      )
    self$deconv2 <-
      layer_conv_2d_transpose(
        filters = 32,
        kernel_size = 3,
        strides = 2,
        padding = "same",
        activation = "relu"
      )
    self$deconv3 <-
      layer_conv_2d_transpose(
        filters = 1,
        kernel_size = 3,
        strides = 1,
        padding = "same"
      )
    
    function (x, mask = NULL) {
      x %>%
        self$dense() %>%
        self$reshape() %>%
        self$deconv1() %>%
        self$deconv2() %>%
        self$deconv3()
    }
  })
}

reparameterize <- function(mean, logvar) {
  eps <- k_random_normal(shape = mean$shape, dtype = tf$float64)
  eps * k_exp(logvar * 0.5) + mean
}


# Loss and optimizer ------------------------------------------------------

normal_loglik <- function(sample, mean, logvar, reduce_axis = 2) {
  loglik <- k_constant(0.5, dtype = tf$float64) * 
    (k_log(2 * k_constant(pi, dtype = tf$float64)) +
     logvar +
     k_exp(-logvar) * (sample - mean) ^ 2)
  - k_sum(loglik, axis = reduce_axis)
}

optimizer <- tf$train$AdamOptimizer(1e-4)



# Output utilities --------------------------------------------------------

num_examples_to_generate <- 64

random_vector_for_generation <-
  k_random_normal(shape = list(num_examples_to_generate, latent_dim),
                  dtype = tf$float64)

generate_random_clothes <- function(epoch) {
  predictions <-
    decoder(random_vector_for_generation) %>% tf$nn$sigmoid()
  png(paste0("cvae_clothes_epoch_", epoch, ".png"))
  par(mfcol = c(8, 8))
  par(mar = c(0.5, 0.5, 0.5, 0.5),
      xaxs = 'i',
      yaxs = 'i')
  for (i in 1:64) {
    img <- predictions[i, , , 1]
    img <- t(apply(img, 2, rev))
    image(
      1:28,
      1:28,
      img * 127.5 + 127.5,
      col = gray((0:255) / 255),
      xaxt = 'n',
      yaxt = 'n'
    )
  }
  dev.off()
}

show_latent_space <- function(epoch) {
  iter <- make_iterator_one_shot(test_dataset)
  x <-  iterator_get_next(iter)
  x_test_encoded <- encoder(x)[[1]]
  x_test_encoded %>%
    as.matrix() %>%
    as.data.frame() %>%
    mutate(class = class_names[fashion$test$y + 1]) %>%
    ggplot(aes(x = V1, y = V2, colour = class)) + geom_point() +
    theme(aspect.ratio = 1) +
    theme(plot.margin = unit(c(0, 0, 0, 0), "null")) +
    theme(panel.spacing = unit(c(0, 0, 0, 0), "null"))
  
  ggsave(
    paste0("cvae_latentspace_epoch_", epoch, ".png"),
    width = 10,
    height = 10,
    units = "cm"
  )
}


show_grid <- function(epoch) {
  png(paste0("cvae_grid_epoch_", epoch, ".png"))
  par(mar = c(0.5, 0.5, 0.5, 0.5),
      xaxs = 'i',
      yaxs = 'i')
  n <- 16
  img_size <- 28
  
  grid_x <- seq(-4, 4, length.out = n)
  grid_y <- seq(-4, 4, length.out = n)
  rows <- NULL
  
  for (i in 1:length(grid_x)) {
    column <- NULL
    for (j in 1:length(grid_y)) {
      z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
      column <-
        rbind(column,
              (decoder(z_sample) %>% tf$nn$sigmoid() %>% as.numeric()) %>% matrix(ncol = img_size))
    }
    rows <- cbind(rows, column)
  }
  rows %>% as.raster() %>% plot()
  dev.off()
}


# Training loop -----------------------------------------------------------

num_epochs <- 50

encoder <- encoder_model()
decoder <- decoder_model()

checkpoint_dir <- "./checkpoints_fashion_cvae"
checkpoint_prefix <- file.path(checkpoint_dir, "ckpt")
checkpoint <-
  tf$train$Checkpoint(optimizer = optimizer,
                      encoder = encoder,
                      decoder = decoder)

generate_random_clothes(0)
show_latent_space(0)
show_grid(0)


for (epoch in seq_len(num_epochs)) {
  iter <- make_iterator_one_shot(train_dataset)
  
  total_loss <- 0
  logpx_z_total <- 0
  logpz_total <- 0
  logqz_x_total <- 0
  
  until_out_of_range({
    x <-  iterator_get_next(iter)
    
    with(tf$GradientTape(persistent = TRUE) %as% tape, {
      
      c(mean, logvar) %<-% encoder(x)
      z <- reparameterize(mean, logvar)
      preds <- decoder(z)
      
      crossentropy_loss <-
        tf$nn$sigmoid_cross_entropy_with_logits(logits = preds, labels = x)
      logpx_z <-
        -k_sum(crossentropy_loss)
      logpz <-
        normal_loglik(z,
                      k_constant(0, dtype = tf$float64),
                      k_constant(0, dtype = tf$float64))
      logqz_x <- normal_loglik(z, mean, logvar)
      loss <- -k_mean(logpx_z + logpz - logqz_x)
      
    })
    
    total_loss <- total_loss + loss
    logpx_z_total <- tf$reduce_mean(logpx_z) + logpx_z_total
    logpz_total <- tf$reduce_mean(logpz) + logpz_total
    logqz_x_total <- tf$reduce_mean(logqz_x) + logqz_x_total
    
    encoder_gradients <- tape$gradient(loss, encoder$variables)
    decoder_gradients <- tape$gradient(loss, decoder$variables)
    
    optimizer$apply_gradients(purrr::transpose(list(
      encoder_gradients, encoder$variables
    )),
    global_step = tf$train$get_or_create_global_step())
    optimizer$apply_gradients(purrr::transpose(list(
      decoder_gradients, decoder$variables
    )),
    global_step = tf$train$get_or_create_global_step())
    
  })
  
  checkpoint$save(file_prefix = checkpoint_prefix)
  
  cat(
    glue(
      "Losses (epoch): {epoch}:",
      "  {(as.numeric(logpx_z_total)/batches_per_epoch) %>% round(2)} logpx_z_total,",
      "  {(as.numeric(logpz_total)/batches_per_epoch) %>% round(2)} logpz_total,",
      "  {(as.numeric(logqz_x_total)/batches_per_epoch) %>% round(2)} logqz_x_total,",
      "  {(as.numeric(total_loss)/batches_per_epoch) %>% round(2)} total"
    ),
    "\n"
  )
  
  if (epoch %% 10 == 0) {
    generate_random_clothes(epoch)
    show_latent_space(epoch)
    show_grid(epoch)
  }
}

