#' Deep Dreaming in Keras.
#' 
#' Note: It is preferable to run this script on GPU, for speed.
#' Example results: http://i.imgur.com/FX6ROg9.jpg
#'

library(keras)
library(tensorflow)
library(purrr)
library(R6)
K <- backend()

# Function Definitions ----------------------------------------------------

preprocess_image <- function(image_path, height, width){
  image_load(image_path, target_size = c(height, width)) %>%
    image_to_array() %>%
    array_reshape(dim = c(1, dim(.))) %>%
    imagenet_preprocess_input()
}

deprocess_image <- function(x){
  x <- x[1,,,]
  
  # Remove zero-center by mean pixel
  x[,,1] <- x[,,1] + 103.939
  x[,,2] <- x[,,2] + 116.779
  x[,,3] <- x[,,3] + 123.68
  
  # 'BGR'->'RGB'
  x <- x[,,c(3,2,1)]
  
  # Clip to interval 0, 255
  x[x > 255] <- 255
  x[x < 0] <- 0
  x[] <- as.integer(x)/255
  x
}

# Calculates the total variation loss
  # See https://en.wikipedia.org/wiki/Total_variation_denoising 
  # for further explanation
total_variation_loss <- function(x, h, w){
  
  y_ij  <- x[,1:(h - 1L), 1:(w - 1L),]
  y_i1j <- x[,2:(h), 1:(w - 1L),]
  y_ij1 <- x[,1:(h - 1L), 2:(w),]
  
  a <- K$square(y_ij - y_i1j)
  b <- K$square(y_ij - y_ij1)
  K$sum(K$pow(a + b, 1.25))
}


# Parameters --------------------------------------------------------

# Some interesting parameter groupings we found
saved_settings = list(
  bad_trip = list(
    features = list(
      block4_conv1 = 0.05,
      block4_conv2 = 0.01,
      block4_conv3 = 0.01
    ),
    continuity = 0.1,
    dream_l2 = 0.8,
    jitter =  5
  ),
  dreamy = list(
    features = list(
      block5_conv1 = 0.05,
      block5_conv2 = 0.02
    ),
    continuity = 0.1,
    dream_l2 = 0.02,
    jitter = 0
  )
)

# The settings to be used in this experiment
img_height <- 600L
img_width <- 600L
img_size <- c(img_height, img_width, 3)
settings <- saved_settings$dreamy
image <- preprocess_image("deep_dream.jpg", img_height, img_width)

# Model Definition --------------------------------------------------------

# This will contain our generated image
dream <- layer_input(batch_shape = c(1, img_size))

# Build the VGG16 network with our placeholder
  # The model will be loaded with pre-trained ImageNet weights
model <- application_vgg16(input_tensor = dream, weights = "imagenet",
                           include_top = FALSE)

# Get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict <- model$layers
names(layer_dict) <- map_chr(layer_dict ,~.x$name)

# Define the loss
loss <- tf$Variable(0.0)
for(layer_name in names(settings$features)){
  
  # Add the L2 norm of the features of a layer to the loss
  coeff <- settings$features[[layer_name]]
  x <- layer_dict[[layer_name]]$output
  out_shape <- layer_dict[[layer_name]]$output_shape %>% unlist()
  
  # Avoid border artifacts by only involving non-border pixels in the loss
  loss <- loss - 
    coeff*K$sum(K$square(x[,4:(out_shape[2] - 1), 4:(out_shape[3] - 1),])) / 
    prod(out_shape[-1])
}

# Add continuity loss (gives image local coherence, can result in an artful blur)
loss <- loss + settings$continuity*
  total_variation_loss(x = dream, img_height, img_width)/
  prod(img_size)

# Add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
  # Note that the loss can be further modified to achieve new effects
loss <- loss + settings$dream_l2*K$sum(K$square(dream))/prod(img_size)

# Compute the gradients of the dream wrt the loss
grads <- K$gradients(loss, dream)[[1]] 

f_outputs <- K$`function`(list(dream), list(loss,grads))

eval_loss_and_grads <- function(image){
  image <- array_reshape(image, c(1, img_size))
  outs <- f_outputs(list(image))
  list(
    loss_value = outs[[1]],
    grad_values = array_reshape(outs[[2]], dim = length(outs[[2]]))
  )
}

# Loss and gradients evaluator:
  # This Evaluator class makes it possible to compute loss and 
  # gradients in one pass while retrieving them via two separate 
  # functions, "loss" and "grads". This is done because 
  # scipy.optimize requires separate functions for loss and 
  # gradients, but computing them separately would be inefficient.
Evaluator <- R6Class(
  "Evaluator",
  public = list(
    
    loss_value = NULL,
    grad_values = NULL,
    
    initialize = function() {
      self$loss_value <- NULL
      self$grad_values <- NULL
    },
    
    loss = function(x){
      loss_and_grad <- eval_loss_and_grads(x)
      self$loss_value <- loss_and_grad$loss_value
      self$grad_values <- loss_and_grad$grad_values
      self$loss_value
    },
    
    grads = function(x){
      grad_values <- self$grad_values
      self$loss_value <- NULL
      self$grad_values <- NULL
      grad_values
    }      
  )
)

evaluator <- Evaluator$new()

# Run optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the loss
for(i in 1:5){
  
  # Add random jitter to initial image
  random_jitter <- settings$jitter*2*(runif(prod(img_size)) - 0.5) %>%
    array(dim = c(1, img_size))
  image <- image + random_jitter

  # Run L-BFGS
  opt <- optim(
    array_reshape(image, dim = length(image)), fn = evaluator$loss, gr = evaluator$grads, 
    method = "L-BFGS-B",
    control = list(maxit = 2)
    )
  
  # Print loss value
  print(opt$value)
  
  # Decode the image
  image <- opt$par
  image <- array_reshape(image, c(1, img_size))
  image <- image - random_jitter

  # Plot
  im <- deprocess_image(image)
  plot(as.raster(im)) 
}
