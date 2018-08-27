#' This is the companion code to the post
#' "Generating digits with Keras and TensorFlow eager execution"
#' on the TensorFlow for R blog.
#'
#' https://blogs.rstudio.com/tensorflow/posts/2018-08-26-eager-dcgan/


library(keras)
use_implementation("tensorflow")
use_session_with_seed(7777, disable_gpu = FALSE, disable_parallel_cpu = FALSE)
library(tensorflow)
tfe_enable_eager_execution(device_policy = "silent")

library(tfdatasets)


mnist <- dataset_mnist()
c(train_images, train_labels) %<-% mnist$train

train_images <- train_images %>%
  k_expand_dims() %>%
  k_cast(dtype = "float32")

train_images <- (train_images - 127.5) / 127.5

buffer_size <- 60000
batch_size <- 256
batches_per_epoch <- (buffer_size / batch_size) %>% round()

train_dataset <- tensor_slices_dataset(train_images) %>%
  dataset_shuffle(buffer_size) %>%
  dataset_batch(batch_size)

generator <-
  function(name = NULL) {
    keras_model_custom(name = name, function(self) {
      self$fc1 <- layer_dense(units = 7 * 7 * 64, use_bias = FALSE)
      self$batchnorm1 <- layer_batch_normalization()
      self$leaky_relu1 <- layer_activation_leaky_relu()
      
      self$conv1 <-
        layer_conv_2d_transpose(
          filters = 64,
          kernel_size = c(5, 5),
          strides = c(1, 1),
          padding = "same",
          use_bias = FALSE
        )
      self$batchnorm2 <- layer_batch_normalization()
      self$leaky_relu2 <- layer_activation_leaky_relu()
      
      self$conv2 <-
        layer_conv_2d_transpose(
          filters = 32,
          kernel_size = c(5, 5),
          strides = c(2, 2),
          padding = "same",
          use_bias = FALSE
        )
      self$batchnorm3 <- layer_batch_normalization()
      self$leaky_relu3 <- layer_activation_leaky_relu()
      
      self$conv3 <-
        layer_conv_2d_transpose(
          filters = 1,
          kernel_size = c(5, 5),
          strides = c(2, 2),
          padding = "same",
          use_bias = FALSE,
          activation = "tanh"
        )
      
      function(inputs,
               mask = NULL,
               training = TRUE) {
        self$fc1(inputs) %>%
          self$batchnorm1(training = training) %>%
          self$leaky_relu1() %>%
          k_reshape(shape = c(-1, 7, 7, 64)) %>%
          
          self$conv1() %>%
          self$batchnorm2(training = training) %>%
          self$leaky_relu2() %>%
          
          self$conv2() %>%
          self$batchnorm3(training = training) %>%
          self$leaky_relu3() %>%
          
          self$conv3()
      }
    })
  }

discriminator <-
  function(name = NULL) {
    keras_model_custom(name = name, function(self) {
      self$conv1 <- layer_conv_2d(
        filters = 64,
        kernel_size = c(5, 5),
        strides = c(2, 2),
        padding = "same"
      )
      self$leaky_relu1 <- layer_activation_leaky_relu()
      self$dropout <- layer_dropout(rate = 0.3)
      
      self$conv2 <-
        layer_conv_2d(
          filters = 128,
          kernel_size = c(5, 5),
          strides = c(2, 2),
          padding = "same"
        )
      self$leaky_relu2 <- layer_activation_leaky_relu()
      self$flatten <- layer_flatten()
      self$fc1 <- layer_dense(units = 1)
      
      function(inputs,
               mask = NULL,
               training = TRUE) {
        inputs %>% self$conv1() %>%
          self$leaky_relu1() %>%
          self$dropout(training = training) %>%
          self$conv2() %>%
          self$leaky_relu2() %>%
          self$flatten() %>%
          self$fc1()
        
      }
    })
  }

generator <- generator()
discriminator <- discriminator()

generator$call = tf$contrib$eager$defun(generator$call)
discriminator$call = tf$contrib$eager$defun(discriminator$call)

discriminator_loss <- function(real_output, generated_output) {
  real_loss <-
    tf$losses$sigmoid_cross_entropy(multi_class_labels = k_ones_like(real_output),
                                    logits = real_output)
  generated_loss <-
    tf$losses$sigmoid_cross_entropy(multi_class_labels = k_zeros_like(generated_output),
                                    logits = generated_output)
  real_loss + generated_loss
}

generator_loss <- function(generated_output) {
  tf$losses$sigmoid_cross_entropy(tf$ones_like(generated_output), generated_output)
}

discriminator_optimizer <- tf$train$AdamOptimizer(1e-4)
generator_optimizer <- tf$train$AdamOptimizer(1e-4)

num_epochs <- 150
noise_dim <- 100
num_examples_to_generate <- 25L

random_vector_for_generation <-
  k_random_normal(c(num_examples_to_generate,
                    noise_dim))

generate_and_save_images <- function(model, epoch, test_input) {
  predictions <- model(test_input, training = FALSE)
  png(paste0("images_epoch_", epoch, ".png"))
  par(mfcol = c(5, 5))
  par(mar = c(0.5, 0.5, 0.5, 0.5),
      xaxs = 'i',
      yaxs = 'i')
  for (i in 1:25) {
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

train <- function(dataset, epochs, noise_dim) {
  for (epoch in seq_len(num_epochs)) {
    start <- Sys.time()
    total_loss_gen <- 0
    total_loss_disc <- 0
    iter <- make_iterator_one_shot(train_dataset)
    
    until_out_of_range({
      batch <- iterator_get_next(iter)
      noise <- k_random_normal(c(batch_size, noise_dim))
      with(tf$GradientTape() %as% gen_tape, {
        with(tf$GradientTape() %as% disc_tape, {
          generated_images <- generator(noise)
          disc_real_output <- discriminator(batch, training = TRUE)
          disc_generated_output <-
            discriminator(generated_images, training = TRUE)
          gen_loss <- generator_loss(disc_generated_output)
          disc_loss <-
            discriminator_loss(disc_real_output, disc_generated_output)
        })
      })
      
      gradients_of_generator <-
        gen_tape$gradient(gen_loss, generator$variables)
      gradients_of_discriminator <-
        disc_tape$gradient(disc_loss, discriminator$variables)
      
      generator_optimizer$apply_gradients(purrr::transpose(list(
        gradients_of_generator, generator$variables
      )))
      discriminator_optimizer$apply_gradients(purrr::transpose(
        list(gradients_of_discriminator, discriminator$variables)
      ))
      
      total_loss_gen <- total_loss_gen + gen_loss
      total_loss_disc <- total_loss_disc + disc_loss
      
    })
    
    cat("Time for epoch ", epoch, ": ", Sys.time() - start, "\n")
    cat("Generator loss: ",
        total_loss_gen$numpy() / batches_per_epoch,
        "\n")
    cat("Discriminator loss: ",
        total_loss_disc$numpy() / batches_per_epoch,
        "\n\n")
    if (epoch %% 10 == 0)
      generate_and_save_images(generator,
                               epoch,
                               random_vector_for_generation)
    
  }
}

train(train_dataset, num_epochs, noise_dim)
