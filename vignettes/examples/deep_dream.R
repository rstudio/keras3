#' Deep Dreaming in Keras.
#' 
#' Note: It is preferable to run this script on GPU, for speed.
#' Example results: http://i.imgur.com/FX6ROg9.jpg
#'

library(keras)
library(tensorflow)
library(purrr)

# Function Definitions ----------------------------------------------------

preprocess_image <- function(image_path){
  image_load(image_path) %>%
    image_to_array() %>%
    array_reshape(dim = c(1, dim(.))) %>%
    inception_v3_preprocess_input()
}

deprocess_image <- function(x){
  x <- x[1,,,]
  
  # Remove zero-center by mean pixel
  x <- x/2.
  x <- x + 0.5
  x <- x * 255
  
  # 'BGR'->'RGB'
  x <- x[,,c(3,2,1)]
  
  # Clip to interval 0, 255
  x[x > 255] <- 255
  x[x < 0] <- 0
  x[] <- as.integer(x)/255
  x
}

# Parameters --------------------------------------------------------

# Some interesting parameter groupings we found
settings <- list(
  features = list(
    mixed2 = 0.2,
    mixed3 = 0.5,
    mixed4 = 2.,
    mixed5 = 1.5
  )
)

# The settings to be used in this experiment
image <- preprocess_image("deep_dream.jpg")

# Model Definition --------------------------------------------------------

k_set_learning_phase(0)

# Build the InceptionV3 network with our placeholder.
# The model will be loaded with pre-trained ImageNet weights.
model <- application_inception_v3(weights = "imagenet", include_top = FALSE)

# This will contain our generated image
dream <- model$input

# Define the loss
loss <- k_variable(0.0)
for(layer_name in names(settings$features)){
  
  # Add the L2 norm of the features of a layer to the loss
  coeff <- settings$features[[layer_name]]
  x <- model$get_layer(layer_name)$output
  scaling <- k_prod(k_cast(k_shape(x), 'float32'))
  
  # Avoid border artifacts by only involving non-border pixels in the loss
  loss <- loss + coeff*k_sum(k_square(x)) / scaling
}


# Compute the gradients of the dream wrt the loss
grads <- k_gradients(loss, dream)[[1]] 

# Normalize gradients.
grads <- grads / k_maximum(k_mean(k_abs(grads)), k_epsilon())

# Set up function to retrieve the value
# of the loss and gradients given an input image.
fetch_loss_and_grads <- k_function(list(dream), list(loss,grads))

eval_loss_and_grads <- function(image){
  outs <- fetch_loss_and_grads(list(image))
  list(
    loss_value = outs[[1]],
    grad_values = outs[[2]]
  )
}


gradient_ascent <- function(x, iterations, step, max_loss = NULL) {
  for (i in 1:iterations) {
    out <- eval_loss_and_grads(x)
    if (!is.null(max_loss) & out$loss_value > max_loss) {
      break
    } 
    print(paste("Loss value at", i, ':', out$loss_value))
    x <- x + step * out$grad_values
  } 
  x
}


# Playing with these hyperparameters will also allow you to achieve new effects
step <- 0.01  # Gradient ascent step size
num_octave <- 3  # Number of scales at which to run gradient ascent
octave_scale <- 1.4  # Size ratio between scales
iterations <- 20  # Number of ascent steps per scale
max_loss <- 10

original_shape <- dim(image)[-c(1, 4)]
successive_shapes <- list(original_shape)

for (i in 1:num_octave) {
  successive_shapes[[i+1]] <- as.integer(original_shape/octave_scale^i)
}
successive_shapes <- rev(successive_shapes)

original_image <- image
shrunk_original_img <- image_array_resize(
  image, successive_shapes[[1]][1], successive_shapes[[1]][2]
  )

for (shp in successive_shapes) {
  
  image <- image_array_resize(image, shp[1], shp[2])
  image <- gradient_ascent(image, iterations, step, max_loss)
  upscaled_shrunk_original_img <- image_array_resize(shrunk_original_img, shp[1], shp[2])
  same_size_original <- image_array_resize(original_image, shp[1], shp[2])
  lost_detail <- same_size_original - upscaled_shrunk_original_img
  
  image <- image + lost_detail
  shrunk_original_img <- image_array_resize(original_image, shp[1], shp[2])
}


plot(as.raster(deprocess_image(image)))

