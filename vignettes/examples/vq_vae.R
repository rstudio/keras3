#' This is the companion code to the post
#' "Discrete Representation Learning with VQ-VAE and TensorFlow Probability"
#' on the TensorFlow for R blog.
#'
#' https://blogs.rstudio.com/tensorflow/posts/2019-01-24-vq-vae/

library(keras)
use_implementation("tensorflow")
library(tensorflow)
tfe_enable_eager_execution(device_policy = "silent")

use_session_with_seed(7778,
                      disable_gpu = FALSE,
                      disable_parallel_cpu = FALSE)

tfp <- import("tensorflow_probability")
tfd <- tfp$distributions

library(tfdatasets)
library(dplyr)
library(glue)
library(curry)

moving_averages <- tf$python$training$moving_averages


# Utilities --------------------------------------------------------

visualize_images <-
  function(dataset,
           epoch,
           reconstructed_images,
           random_images) {
    write_png(dataset, epoch, "reconstruction", reconstructed_images)
    write_png(dataset, epoch, "random", random_images)
    
  }

write_png <- function(dataset, epoch, desc, images) {
  png(paste0(dataset, "_epoch_", epoch, "_", desc, ".png"))
  par(mfcol = c(8, 8))
  par(mar = c(0.5, 0.5, 0.5, 0.5),
      xaxs = 'i',
      yaxs = 'i')
  for (i in 1:64) {
    img <- images[i, , , 1]
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


# Setup and preprocessing -------------------------------------------------

np <- import("numpy")

# download from: https://github.com/rois-codh/kmnist
kuzushiji <- np$load("kmnist-train-imgs.npz")
kuzushiji <- kuzushiji$get("arr_0")

train_images <- kuzushiji %>%
  k_expand_dims() %>%
  k_cast(dtype = "float32")
train_images <- train_images %>% `/`(255)

buffer_size <- 60000
batch_size <- 64
num_examples_to_generate <- batch_size

batches_per_epoch <- buffer_size / batch_size

train_dataset <- tensor_slices_dataset(train_images) %>%
  dataset_shuffle(buffer_size) %>%
  dataset_batch(batch_size, drop_remainder = TRUE)

# test
iter <- make_iterator_one_shot(train_dataset)
batch <-  iterator_get_next(iter)
batch %>% dim()

# Params ------------------------------------------------------------------

learning_rate <- 0.001
latent_size <- 1
num_codes <- 64L
code_size <- 16L
base_depth <- 32
activation <- "elu"
beta <- 0.25
decay <- 0.99
input_shape <- c(28, 28, 1)

# Models -------------------------------------------------------------------

default_conv <-
  set_defaults(layer_conv_2d, list(padding = "same", activation = activation))
default_deconv <-
  set_defaults(layer_conv_2d_transpose,
               list(padding = "same", activation = activation))

# Encoder ------------------------------------------------------------------

encoder_model <- function(name = NULL,
                          code_size) {
  
  keras_model_custom(name = name, function(self) {
    self$conv1 <- default_conv(filters = base_depth, kernel_size = 5)
    self$conv2 <-
      default_conv(filters = base_depth,
                   kernel_size = 5,
                   strides = 2)
    self$conv3 <-
      default_conv(filters = 2 * base_depth, kernel_size = 5)
    self$conv4 <-
      default_conv(
        filters = 2 * base_depth,
        kernel_size = 5,
        strides = 2
      )
    self$conv5 <-
      default_conv(
        filters = 4 * latent_size,
        kernel_size = 7,
        padding = "valid"
      )
    self$flatten <- layer_flatten()
    self$dense <- layer_dense(units = latent_size * code_size)
    self$reshape <-
      layer_reshape(target_shape = c(latent_size, code_size))
    
    function (x, mask = NULL) {
      x %>%
        # output shape:  7 28 28 32
        self$conv1() %>%
        # output shape:  7 14 14 32
        self$conv2() %>%
        # output shape:  7 14 14 64
        self$conv3() %>%
        # output shape:  7 7 7 64
        self$conv4() %>%
        # output shape:  7 1 1 4
        self$conv5() %>%
        # output shape:  7 4
        self$flatten() %>%
        # output shape:  7 16
        self$dense() %>%
        # output shape:  7 1 16
        self$reshape()
    }
    
  })
}


# Decoder ------------------------------------------------------------------

decoder_model <- function(name = NULL,
                          input_size,
                          output_shape) {
  
  keras_model_custom(name = name, function(self) {
    self$reshape1 <- layer_reshape(target_shape = c(1, 1, input_size))
    self$deconv1 <-
      default_deconv(
        filters = 2 * base_depth,
        kernel_size = 7,
        padding = "valid"
      )
    self$deconv2 <-
      default_deconv(filters = 2 * base_depth, kernel_size = 5)
    self$deconv3 <-
      default_deconv(
        filters = 2 * base_depth,
        kernel_size = 5,
        strides = 2
      )
    self$deconv4 <-
      default_deconv(filters = base_depth, kernel_size = 5)
    self$deconv5 <-
      default_deconv(filters = base_depth,
                     kernel_size = 5,
                     strides = 2)
    self$deconv6 <-
      default_deconv(filters = base_depth, kernel_size = 5)
    self$conv1 <-
      default_conv(filters = output_shape[3],
                   kernel_size = 5,
                   activation = "linear")
    
    function (x, mask = NULL) {
      x <- x %>%
        # output shape:  7 1 1 16
        self$reshape1() %>%
        # output shape:  7 7 7 64
        self$deconv1() %>%
        # output shape:  7 7 7 64
        self$deconv2() %>%
        # output shape:  7 14 14 64
        self$deconv3() %>%
        # output shape:  7 14 14 32
        self$deconv4() %>%
        # output shape:  7 28 28 32
        self$deconv5() %>%
        # output shape:  7 28 28 32
        self$deconv6() %>%
        # output shape:  7 28 28 1
        self$conv1()
      tfd$Independent(tfd$Bernoulli(logits = x),
                      reinterpreted_batch_ndims = length(output_shape))
    }
  })
}

# Vector quantizer -------------------------------------------------------------------

vector_quantizer_model <- 
  function(name = NULL, num_codes, code_size) {
    
    keras_model_custom(name = name, function(self) {
      self$num_codes <- num_codes
      self$code_size <- code_size
      self$codebook <- tf$get_variable("codebook",
                                       shape = c(num_codes, code_size),
                                       dtype = tf$float32)
      self$ema_count <- tf$get_variable(
        name = "ema_count",
        shape = c(num_codes),
        initializer = tf$constant_initializer(0),
        trainable = FALSE
      )
      self$ema_means = tf$get_variable(
        name = "ema_means",
        initializer = self$codebook$initialized_value(),
        trainable = FALSE
      )
      
      function (x, mask = NULL) {

        # bs * 1 * num_codes
        distances <- tf$norm(tf$expand_dims(x, axis = 2L) -
                               tf$reshape(self$codebook,
                                          c(
                                            1L, 1L, self$num_codes, self$code_size
                                          )),
                             axis = 3L)
        
        # bs * 1
        assignments <- tf$argmin(distances, axis = 2L)
        
        # bs * 1 * num_codes
        one_hot_assignments <-
          tf$one_hot(assignments, depth = self$num_codes)
        
        # bs * 1 * code_size
        nearest_codebook_entries <- tf$reduce_sum(
          tf$expand_dims(one_hot_assignments,-1L) * # bs, 1, 64, 1
            tf$reshape(self$codebook, c(
              1L, 1L, self$num_codes, self$code_size
            )),
          axis = 2L # 1, 1, 64, 16
        )
        
        list(nearest_codebook_entries, one_hot_assignments)
      }
    })
  }


# Update codebook ------------------------------------------------------

update_ema <- function(vector_quantizer,
                       one_hot_assignments,
                       codes,
                       decay) {
  # shape = 64
  updated_ema_count <- moving_averages$assign_moving_average(
    vector_quantizer$ema_count,
    tf$reduce_sum(one_hot_assignments, axis = c(0L, 1L)),
    decay,
    zero_debias = FALSE
  )
  
  # 64 * 16
  updated_ema_means <- moving_averages$assign_moving_average(
    vector_quantizer$ema_means,
    # selects all assigned values (masking out the others) and sums them up over the batch
    # (will be divided by count later)
    tf$reduce_sum(
      tf$expand_dims(codes, 2L) *
        tf$expand_dims(one_hot_assignments, 3L),
      axis = c(0L, 1L)
    ),
    decay,
    zero_debias = FALSE
  )
  
  # Add small value to avoid dividing by zero
  updated_ema_count <- updated_ema_count + 1e-5
  updated_ema_means <-
    updated_ema_means / tf$expand_dims(updated_ema_count, axis = -1L)
  
  tf$assign(vector_quantizer$codebook, updated_ema_means)
}


# Training setup -----------------------------------------------------------

encoder <- encoder_model(code_size = code_size)
decoder <- decoder_model(input_size = latent_size * code_size,
                         output_shape = input_shape)

vector_quantizer <-
  vector_quantizer_model(num_codes = num_codes, code_size = code_size)

optimizer <- tf$train$AdamOptimizer(learning_rate = learning_rate)

checkpoint_dir <- "./vq_vae_checkpoints"

checkpoint_prefix <- file.path(checkpoint_dir, "ckpt")
checkpoint <-
  tf$train$Checkpoint(
    optimizer = optimizer,
    encoder = encoder,
    decoder = decoder,
    vector_quantizer_model = vector_quantizer
  )

checkpoint$save(file_prefix = checkpoint_prefix)

# Training loop -----------------------------------------------------------

num_epochs <- 20

for (epoch in seq_len(num_epochs)) {
  
  iter <- make_iterator_one_shot(train_dataset)
  
  total_loss <- 0
  reconstruction_loss_total <- 0
  commitment_loss_total <- 0
  prior_loss_total <- 0
  
  until_out_of_range({
    
    x <-  iterator_get_next(iter)
    
    with(tf$GradientTape(persistent = TRUE) %as% tape, {
      
      codes <- encoder(x)
      c(nearest_codebook_entries, one_hot_assignments) %<-% vector_quantizer(codes)
      codes_straight_through <- codes + tf$stop_gradient(nearest_codebook_entries - codes)
      decoder_distribution <- decoder(codes_straight_through)
      
      reconstruction_loss <-
        -tf$reduce_mean(decoder_distribution$log_prob(x))
      
      commitment_loss <- tf$reduce_mean(tf$square(codes - tf$stop_gradient(nearest_codebook_entries)))
      
      prior_dist <- tfd$Multinomial(total_count = 1,
                                    logits = tf$zeros(c(latent_size, num_codes)))
      prior_loss <- -tf$reduce_mean(tf$reduce_sum(prior_dist$log_prob(one_hot_assignments), 1L))
      
      loss <-
        reconstruction_loss + beta * commitment_loss + prior_loss
      
    })
    
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
    
    update_ema(vector_quantizer,
               one_hot_assignments,
               codes,
               decay)
    
    total_loss <- total_loss + loss
    reconstruction_loss_total <-
      reconstruction_loss_total + reconstruction_loss
    commitment_loss_total <- commitment_loss_total + commitment_loss
    prior_loss_total <- prior_loss_total + prior_loss
    
  })
  
  checkpoint$save(file_prefix = checkpoint_prefix)
  
  cat(
    glue(
      "Loss (epoch): {epoch}:",
      "  {(as.numeric(total_loss)/trunc(buffer_size/batch_size)) %>% round(4)} loss",
      "  {(as.numeric(reconstruction_loss_total)/trunc(buffer_size/batch_size)) %>% round(4)} reconstruction_loss",
      "  {(as.numeric(commitment_loss_total)/trunc(buffer_size/batch_size)) %>% round(4)} commitment_loss",
      "  {(as.numeric(prior_loss_total)/trunc(buffer_size/batch_size)) %>% round(4)} prior_loss",
      
    ),
    "\n"
  )
  
  # display example images (choose your frequency)
  if (TRUE) {
    reconstructed_images <- decoder_distribution$mean()
    # (64, 1, 16)
    prior_samples <- tf$reduce_sum(
      # selects one of the codes (masking out 63 of 64 codes)
      # (bs, 1, 64, 1)
      tf$expand_dims(prior_dist$sample(num_examples_to_generate),-1L) *
        # (1, 1, 64, 16)
        tf$reshape(vector_quantizer$codebook,
                   c(1L, 1L, num_codes, code_size)),
      axis = 2L
    )
    decoded_distribution_given_random_prior <-
      decoder(prior_samples)
    random_images <- decoded_distribution_given_random_prior$mean()
    visualize_images("k", epoch, reconstructed_images, random_images)
  }
}
