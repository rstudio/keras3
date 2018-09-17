#' This is the companion code to the post 
#' "Image-to-image translation with Pix2Pix: An implementation using Keras and eager execution"
#' on the TensorFlow for R blog.
#' 
#' https://blogs.rstudio.com/tensorflow/posts/2018-09-20-eager-pix2pix

library(keras)
use_implementation("tensorflow")

library(tensorflow)

tfe_enable_eager_execution(device_policy = "silent")

library(tfdatasets)
library(purrr)

restore <- TRUE

data_dir <- "facades"

buffer_size <- 400
batch_size <- 1
batches_per_epoch <- buffer_size / batch_size
img_width <- 256L
img_height <- 256L

load_image <- function(image_file, is_train) {

  image <- tf$read_file(image_file)
  image <- tf$image$decode_jpeg(image)
  
  w <- as.integer(k_shape(image)[2])
  w2 <- as.integer(w / 2L)
  real_image <- image[ , 1L:w2, ]
  input_image <- image[ , (w2 + 1L):w, ]
  
  input_image <- k_cast(input_image, tf$float32)
  real_image <- k_cast(real_image, tf$float32)

  if (is_train) {
      input_image <-
      tf$image$resize_images(input_image,
                             c(286L, 286L),
                             align_corners = TRUE,
                             method = 2)
    real_image <- tf$image$resize_images(real_image,
                                         c(286L, 286L),
                                         align_corners = TRUE,
                                         method = 2)
    
    stacked_image <-
      k_stack(list(input_image, real_image), axis = 1)
    cropped_image <-
      tf$random_crop(stacked_image, size = c(2L, img_height, img_width, 3L))
    c(input_image, real_image) %<-% list(cropped_image[1, , , ], cropped_image[2, , , ])
    
    if (runif(1) > 0.5) {
      input_image <- tf$image$flip_left_right(input_image)
      real_image <- tf$image$flip_left_right(real_image)
    }
  } else {
    input_image <-
      tf$image$resize_images(
        input_image,
        size = c(img_height, img_width),
        align_corners = TRUE,
        method = 2
      )
    real_image <-
      tf$image$resize_images(
        real_image,
        size = c(img_height, img_width),
        align_corners = TRUE,
        method = 2
      )
  }
  
  input_image <- (input_image / 127.5) - 1
  real_image <- (real_image / 127.5) - 1
  
  list(input_image, real_image)
}

train_dataset <-
  tf$data$Dataset$list_files(file.path(data_dir, "train/*.jpg")) %>%
  dataset_shuffle(buffer_size) %>%
  dataset_map(function(image)
    tf$py_func(load_image, list(image, TRUE), list(tf$float32, tf$float32))) %>%
  dataset_batch(batch_size)

test_dataset <-
  tf$data$Dataset$list_files(file.path(data_dir, "test/*.jpg")) %>%
  dataset_map(function(image)
    tf$py_func(load_image, list(image, TRUE), list(tf$float32, tf$float32))) %>%
  dataset_batch(batch_size)


downsample <- function(filters,
                       size,
                       apply_batchnorm = TRUE,
                       name = "downsample") {
  keras_model_custom(name = name, function(self) {
    self$apply_batchnorm <- apply_batchnorm
    self$conv1 <- layer_conv_2d(
      filters = filters,
      kernel_size = size,
      strides = 2,
      padding = 'same',
      kernel_initializer = initializer_random_normal(0, 0.2),
      use_bias = FALSE
    )
    if (self$apply_batchnorm) {
      self$batchnorm <- layer_batch_normalization()
    }
    
    function(x,
             mask = NULL,
             training = TRUE) {
      x <- self$conv1(x)
      if (self$apply_batchnorm) {
        x %>% self$batchnorm(training = training)
      }
      cat("downsample (generator) output: ", x$shape$as_list(), "\n")
      x %>% layer_activation_leaky_relu()
    }
    
  })
}

upsample <- function(filters,
                     size,
                     apply_dropout = FALSE,
                     name = "upsample") {
  keras_model_custom(name = NULL, function(self) {
    self$apply_dropout <- apply_dropout
    self$up_conv <- layer_conv_2d_transpose(
      filters = filters,
      kernel_size = size,
      strides = 2,
      padding = "same",
      kernel_initializer = initializer_random_normal(),
      use_bias = FALSE
    )
    self$batchnorm <- layer_batch_normalization()
    if (self$apply_dropout) {
      self$dropout <- layer_dropout(rate = 0.5)
    }
    function(xs,
             mask = NULL,
             training = TRUE) {
      c(x1, x2) %<-% xs
      x <- self$up_conv(x1) %>% self$batchnorm(training = training)
      if (self$apply_dropout) {
        x %>% self$dropout(training = training)
      }
      x %>% layer_activation("relu")
      concat <- k_concatenate(list(x, x2))
      cat("upsample (generator) output: ", concat$shape$as_list(), "\n")
      concat
    }
  })
}

generator <- function(name = "generator") {
  keras_model_custom(name = name, function(self) {
    self$down1 <- downsample(64, 4, apply_batchnorm = FALSE)
    self$down2 <- downsample(128, 4)
    self$down3 <- downsample(256, 4)
    self$down4 <- downsample(512, 4)
    self$down5 <- downsample(512, 4)
    self$down6 <- downsample(512, 4)
    self$down7 <- downsample(512, 4)
    self$down8 <- downsample(512, 4)
    
    self$up1 <- upsample(512, 4, apply_dropout = TRUE)
    self$up2 <- upsample(512, 4, apply_dropout = TRUE)
    self$up3 <- upsample(512, 4, apply_dropout = TRUE)
    self$up4 <- upsample(512, 4)
    self$up5 <- upsample(256, 4)
    self$up6 <- upsample(128, 4)
    self$up7 <- upsample(64, 4)
    self$last <- layer_conv_2d_transpose(
      filters = 3,
      kernel_size = 4,
      strides = 2,
      padding = "same",
      kernel_initializer = initializer_random_normal(0, 0.2),
      activation = "tanh"
    )
    
    function(x,
             mask = NULL,
             training = TRUE) {
      # x shape == (bs, 256, 256, 3)
      x1 <-
        x %>% self$down1(training = training)  # (bs, 128, 128, 64)
      x2 <- self$down2(x1, training = training) # (bs, 64, 64, 128)
      x3 <- self$down3(x2, training = training) # (bs, 32, 32, 256)
      x4 <- self$down4(x3, training = training) # (bs, 16, 16, 512)
      x5 <- self$down5(x4, training = training) # (bs, 8, 8, 512)
      x6 <- self$down6(x5, training = training) # (bs, 4, 4, 512)
      x7 <- self$down7(x6, training = training) # (bs, 2, 2, 512)
      x8 <- self$down8(x7, training = training) # (bs, 1, 1, 512)

      x9 <-
        self$up1(list(x8, x7), training = training) # (bs, 2, 2, 1024)
      x10 <-
        self$up2(list(x9, x6), training = training) # (bs, 4, 4, 1024)
      x11 <-
        self$up3(list(x10, x5), training = training) # (bs, 8, 8, 1024)
      x12 <-
        self$up4(list(x11, x4), training = training) # (bs, 16, 16, 1024)
      x13 <-
        self$up5(list(x12, x3), training = training) # (bs, 32, 32, 512)
      x14 <-
        self$up6(list(x13, x2), training = training) # (bs, 64, 64, 256)
      x15 <-
        self$up7(list(x14, x1), training = training) # (bs, 128, 128, 128)
      x16 <- self$last(x15) # (bs, 256, 256, 3)
      cat("generator output: ", x16$shape$as_list(), "\n")
      x16
    }
  })
}


disc_downsample <- function(filters,
                            size,
                            apply_batchnorm = TRUE,
                            name = "disc_downsample") {
  keras_model_custom(name = name, function(self) {
    self$apply_batchnorm <- apply_batchnorm
    self$conv1 <- layer_conv_2d(
      filters = filters,
      kernel_size = size,
      strides = 2,
      padding = 'same',
      kernel_initializer = initializer_random_normal(0, 0.2),
      use_bias = FALSE
    )
    if (self$apply_batchnorm) {
      self$batchnorm <- layer_batch_normalization()
    }
    
    function(x,
             mask = NULL,
             training = TRUE) {
      x <- self$conv1(x)
      if (self$apply_batchnorm) {
        x %>% self$batchnorm(training = training)
      }
      x %>% layer_activation_leaky_relu()
    }
    
  })
}

discriminator <- function(name = "discriminator") {
  keras_model_custom(name = name, function(self) {
    self$down1 <- disc_downsample(64, 4, FALSE)
    self$down2 <- disc_downsample(128, 4)
    self$down3 <- disc_downsample(256, 4)
    # we are zero padding here with 1 because we need our shape to
    # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
    self$zero_pad1 <- layer_zero_padding_2d()
    self$conv <- layer_conv_2d(
      filters = 512,
      kernel_size = 4,
      strides = 1,
      kernel_initializer = initializer_random_normal(),
      use_bias = FALSE
    )
    self$batchnorm <- layer_batch_normalization()
    self$zero_pad2 <- layer_zero_padding_2d()
    self$last <- layer_conv_2d(
      filters = 1,
      kernel_size = 4,
      strides = 1,
      kernel_initializer = initializer_random_normal()
    )
    
    function(x,
             y,
             mask = NULL,
             training = TRUE) {
      x <- k_concatenate(list(x, y)) %>% # (bs, 256, 256, channels*2)
        self$down1(training = training) %>% # (bs, 128, 128, 64)
        self$down2(training = training) %>% # (bs, 64, 64, 128)
        self$down3(training = training) %>% # (bs, 32, 32, 256)
        self$zero_pad1() %>% # (bs, 34, 34, 256)
        self$conv() %>% # (bs, 31, 31, 512)
        self$batchnorm(training = training) %>%
        layer_activation_leaky_relu() %>%
        self$zero_pad2() %>% # (bs, 33, 33, 512)
        self$last() # (bs, 30, 30, 1)
      cat("discriminator output: ", x$shape$as_list(), "\n")
      x
    }
  })
  
}

generator <- generator()
discriminator <- discriminator()

generator$call = tf$contrib$eager$defun(generator$call)
discriminator$call = tf$contrib$eager$defun(discriminator$call)

discriminator_loss <- function(real_output, generated_output) {
  real_loss <-
    tf$losses$sigmoid_cross_entropy(multi_class_labels = tf$ones_like(real_output),
                                    logits = real_output)
  generated_loss <-
    tf$losses$sigmoid_cross_entropy(multi_class_labels = tf$zeros_like(generated_output),
                                    logits = generated_output)
  real_loss + generated_loss
}

lambda <- 100
generator_loss <-
  function(disc_judgment, generated_output, target) {
    gan_loss <-
      tf$losses$sigmoid_cross_entropy(tf$ones_like(disc_judgment), disc_judgment)
    l1_loss <- tf$reduce_mean(tf$abs(target - generated_output))
    gan_loss + (lambda * l1_loss)
  }

discriminator_optimizer <- tf$train$AdamOptimizer(2e-4, beta1 = 0.5)
generator_optimizer <- tf$train$AdamOptimizer(2e-4, beta1 = 0.5)

checkpoint_dir <- "./checkpoints_pix2pix"
checkpoint_prefix <- file.path(checkpoint_dir, "ckpt")
checkpoint <-
  tf$train$Checkpoint(
    generator_optimizer = generator_optimizer,
    discriminator_optimizer = discriminator_optimizer,
    generator = generator,
    discriminator = discriminator
  )

generate_images <- function(generator, input, target, id) {
  prediction <- generator(input, training = TRUE)
  png(paste0("pix2pix_", id, ".png"), width = 900, height = 300)
  par(mfcol = c(1, 3))
  par(mar = c(0, 0, 0, 0),
      xaxs = 'i',
      yaxs = 'i')
  input <- input[1, , ,]$numpy() * 0.5 + 0.5
  input[input > 1] <- 1
  input[input < 0] <- 0
  plot(as.raster(input, main = "input image"))
  target <- target[1, , ,]$numpy() * 0.5 + 0.5
  target[target > 1] <- 1
  target[target < 0] <- 0
  plot(as.raster(target, main = "ground truth"))
  prediction <- prediction[1, , ,]$numpy() * 0.5 + 0.5
  prediction[prediction > 1] <- 1
  prediction[prediction < 0] <- 0
  plot(as.raster(prediction, main = "generated"))
  dev.off()
}

train <- function(dataset, num_epochs) {
  for (epoch in 1:num_epochs) {
    total_loss_gen <- 0
    total_loss_disc <- 0
    iter <- make_iterator_one_shot(train_dataset)
    
    until_out_of_range({
      batch <- iterator_get_next(iter)
      input_image <- batch[[1]]
      target <- batch[[2]]
      
      with(tf$GradientTape() %as% gen_tape, {
        with(tf$GradientTape() %as% disc_tape, {
          gen_output <- generator(input_image, training = TRUE)
          disc_real_output <-
            discriminator(input_image, target, training = TRUE)
          disc_generated_output <-
            discriminator(input_image, gen_output, training = TRUE)
          gen_loss <-
            generator_loss(disc_generated_output, gen_output, target)
          disc_loss <-
            discriminator_loss(disc_real_output, disc_generated_output)
          total_loss_gen <- total_loss_gen + gen_loss
          total_loss_disc <- total_loss_disc + disc_loss
        })
      })
      generator_gradients <- gen_tape$gradient(gen_loss,
                                               generator$variables)
      discriminator_gradients <- disc_tape$gradient(disc_loss,
                                                    discriminator$variables)
      
      generator_optimizer$apply_gradients(transpose(list(
        generator_gradients,
        generator$variables
      )))
      discriminator_optimizer$apply_gradients(transpose(
        list(discriminator_gradients,
             discriminator$variables)
      ))
      
    })
    cat("Epoch ", epoch, "\n")
    cat("Generator loss: ",
        total_loss_gen$numpy() / batches_per_epoch,
        "\n")
    cat("Discriminator loss: ",
        total_loss_disc$numpy() / batches_per_epoch,
        "\n\n")
    if (epoch %% 10 == 0) {
      test_iter <- make_iterator_one_shot(test_dataset)
      batch <- iterator_get_next(test_iter)
      input <- batch[[1]]
      target <- batch[[2]]
      generate_images(generator, input, target, paste0("epoch_", i))
    }
    if (epoch %% 10 == 0) {
      checkpoint$save(file_prefix = checkpoint_prefix)
    }
    
  }
}

if (!restore) {
  train(train_dataset, 200)
} 


checkpoint$restore(tf$train$latest_checkpoint(checkpoint_dir))

test_iter <- make_iterator_one_shot(test_dataset)
i <- 1
until_out_of_range({
  batch <- iterator_get_next(test_iter)
  input <- batch[[1]]
  target <- batch[[2]]
  generate_images(generator, input, target, paste0("test_", i))
  i <- i + 1
})



