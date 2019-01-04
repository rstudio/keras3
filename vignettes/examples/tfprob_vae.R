#' This is the companion code to the post
#' "Getting started with TensorFlow Probability from R"
#' on the TensorFlow for R blog.
#'
#' https://blogs.rstudio.com/tensorflow/posts/2019-01-08-getting-started-with-tf-probability/

library(keras)
use_implementation("tensorflow")
library(tensorflow)
tfe_enable_eager_execution(device_policy = "silent")

tfp <- import("tensorflow_probability")
tfd <- tfp$distributions

library(tfdatasets)
library(dplyr)
library(glue)


# Utilities --------------------------------------------------------

num_examples_to_generate <- 64L

generate_random <- function(epoch) {
  decoder_likelihood <-
    decoder(latent_prior$sample(num_examples_to_generate))
  predictions <- decoder_likelihood$mean()
  # change path according to your preferences
  png(file.path("/tmp", paste0("random_epoch_", epoch, ".png")))
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

show_grid <- function(epoch) {
  # change path according to your preferences
  png(file.path("/tmp", paste0("grid_epoch_", epoch, ".png")))
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
      decoder_likelihood <- decoder(k_cast(z_sample, k_floatx()))
      column <-
        rbind(column,
              (decoder_likelihood$mean() %>% as.numeric()) %>% matrix(ncol = img_size))
    }
    rows <- cbind(rows, column)
  }
  rows %>% as.raster() %>% plot()
  dev.off()
}


# Setup and preprocessing -------------------------------------------------

np <- import("numpy")

# assume data have been downloaded from https://github.com/rois-codh/kmnist
# and stored in /tmp
kuzushiji <- np$load("/tmp/kmnist-train-imgs.npz")
kuzushiji <- kuzushiji$get("arr_0")

train_images <- kuzushiji %>%
  k_expand_dims() %>%
  k_cast(dtype = "float32")
train_images <- train_images %>% `/`(255)

buffer_size <- 60000
batch_size <- 256
batches_per_epoch <- buffer_size / batch_size

train_dataset <- tensor_slices_dataset(train_images) %>%
  dataset_shuffle(buffer_size) %>%
  dataset_batch(batch_size)


# Params ------------------------------------------------------------------

latent_dim <- 2
mixture_components <- 16


# Model -------------------------------------------------------------------

# Encoder ------------------------------------------------------------------

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
      x <- x %>%
        self$conv1() %>%
        self$conv2() %>%
        self$flatten() %>%
        self$dense()
      tfd$MultivariateNormalDiag(loc = x[, 1:latent_dim],
                                 scale_diag = tf$nn$softplus(x[, (latent_dim + 1):(2 * latent_dim)] + 1e-5))
    }
  })
}


# Decoder ------------------------------------------------------------------

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
      x <- x %>%
        self$dense() %>%
        self$reshape() %>%
        self$deconv1() %>%
        self$deconv2() %>%
        self$deconv3()
      
      tfd$Independent(tfd$Bernoulli(logits = x),
                      reinterpreted_batch_ndims = 3L)
      
    }
  })
}

# Learnable Prior -------------------------------------------------------------------

learnable_prior_model <-
  function(name = NULL, latent_dim, mixture_components) {
    
    keras_model_custom(name = name, function(self) {
      self$loc <-
        tf$get_variable(
          name = "loc",
          shape = list(mixture_components, latent_dim),
          dtype = tf$float32
        )
      self$raw_scale_diag <- tf$get_variable(
        name = "raw_scale_diag",
        shape = c(mixture_components, latent_dim),
        dtype = tf$float32
      )
      self$mixture_logits <-
        tf$get_variable(
          name = "mixture_logits",
          shape = c(mixture_components),
          dtype = tf$float32
        )
      
      function (x, mask = NULL) {
        tfd$MixtureSameFamily(
          components_distribution = tfd$MultivariateNormalDiag(
            loc = self$loc,
            scale_diag = tf$nn$softplus(self$raw_scale_diag)
          ),
          mixture_distribution = tfd$Categorical(logits = self$mixture_logits)
        )
      }
    })
  }


# Loss and optimizer ------------------------------------------------------

compute_kl_loss <-
  function(latent_prior,
           approx_posterior,
           approx_posterior_sample) {
    kl_div <- approx_posterior$log_prob(approx_posterior_sample) - latent_prior$log_prob(approx_posterior_sample)
    avg_kl_div <- tf$reduce_mean(kl_div)
    avg_kl_div
  }


global_step <- tf$train$get_or_create_global_step()
optimizer <- tf$train$AdamOptimizer(1e-4)


# Training loop -----------------------------------------------------------

num_epochs <- 50

encoder <- encoder_model()
decoder <- decoder_model()
latent_prior_model <-
  learnable_prior_model(latent_dim = latent_dim, mixture_components = mixture_components)

# change this according to your preferences
checkpoint_dir <- "/tmp/checkpoints"
checkpoint_prefix <- file.path(checkpoint_dir, "ckpt")
checkpoint <-
  tf$train$Checkpoint(
    optimizer = optimizer,
    global_step = global_step,
    encoder = encoder,
    decoder = decoder,
    latent_prior_model = latent_prior_model
  )

for (epoch in seq_len(num_epochs)) {
  iter <- make_iterator_one_shot(train_dataset)
  
  total_loss <- 0
  total_loss_nll <- 0
  total_loss_kl <- 0
  
  until_out_of_range({
    x <-  iterator_get_next(iter)
    
    with(tf$GradientTape(persistent = TRUE) %as% tape, {
      approx_posterior <- encoder(x)
      
      approx_posterior_sample <- approx_posterior$sample()
      decoder_likelihood <- decoder(approx_posterior_sample)
      
      nll <- -decoder_likelihood$log_prob(x)
      avg_nll <- tf$reduce_mean(nll)
      
      latent_prior <- latent_prior_model(NULL)
      
      kl_loss <-
        compute_kl_loss(latent_prior,
                        approx_posterior,
                        approx_posterior_sample)

      loss <- kl_loss + avg_nll
    })
    
    total_loss <- total_loss + loss
    total_loss_nll <- total_loss_nll + avg_nll
    total_loss_kl <- total_loss_kl + kl_loss
    
    encoder_gradients <- tape$gradient(loss, encoder$variables)
    decoder_gradients <- tape$gradient(loss, decoder$variables)
    prior_gradients <-
      tape$gradient(loss, latent_prior_model$variables)
    
    optimizer$apply_gradients(purrr::transpose(list(
      encoder_gradients, encoder$variables
    )),
    global_step = tf$train$get_or_create_global_step())
    optimizer$apply_gradients(purrr::transpose(list(
      decoder_gradients, decoder$variables
    )),
    global_step = tf$train$get_or_create_global_step())
    optimizer$apply_gradients(purrr::transpose(list(
      prior_gradients, latent_prior_model$variables
    )),
    global_step = tf$train$get_or_create_global_step())
    
})
  
  checkpoint$save(file_prefix = checkpoint_prefix)
  
  cat(
    glue(
      "Losses (epoch): {epoch}:",
      "  {(as.numeric(total_loss_nll)/batches_per_epoch) %>% round(4)} nll",
      "  {(as.numeric(total_loss_kl)/batches_per_epoch) %>% round(4)} kl",
      "  {(as.numeric(total_loss)/batches_per_epoch) %>% round(4)} total"
    ),
    "\n"
  )
  
  if (TRUE) {
    generate_random(epoch)
    show_grid(epoch)
  }
}
