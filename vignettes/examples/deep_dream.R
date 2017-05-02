# Deep Dreaming in Keras.
# 
# Run the script with:
# ```
# python deep_dream.py path_to_your_base_image.jpg prefix_for_results
# ```
# e.g.:
# ```
# python deep_dream.py img/mypic.jpg results/dream
# ```
# 
# It is preferable to run this script on GPU, for speed.
# If running on CPU, prefer the TensorFlow backend (much faster).
# 
# Example results: http://i.imgur.com/FX6ROg9.jpg
# '
library(keras)
library(purrr)

# Function Definitions ----------------------------------------------------
vgg16_preprocess_input <- function(x){
  # 'RGB'->'BGR'
  x <- x[,,,c(3,2,1), drop = FALSE]
  # Zero-center by mean pixel
  x[,,,1] <- x[,,,1] - 103.939
  x[,,,2] <- x[,,,2] - 116.779
  x[,,,3] <- x[,,,3] - 123.68
  x
}

preprocess_image <- function(image_path, height, width){
  image_load(image_path, target_size = c(height, width)) %>%
    image_to_array() %>%
    array(dim = c(1, dim(.))) %>%
    vgg16_preprocess_input()
}

deprocess_image <- function(x){
  x <- x[1,,,]
  # Remove zero-center by mean pixel
  x[,,1] <- x[,,1] + 103.939
  x[,,2] <- x[,,2] + 116.779
  x[,,3] <- x[,,3] + 123.68
  # 'BGR'->'RGB'
  x <- x[,,c(3,2,1)]
  # clip to interval 0, 255
  x[x > 255] <- 255
  x[x < 0] <- 0
  x[] <- as.integer(x)
  x
}

# calculates the total variation loss
# https://en.wikipedia.org/wiki/Total_variation_denoising
total_variation_loss <- function(x, w, h){
  
  w1 <- w - 1L
  w2 <- w - 2L
  h1 <- h - 1L
  h2 <- h - 2L
  
  y_ij  <- x[,0:(h - 2L), 0:w2,]
  y_i1j <- x[,1:(h - 1L), 0:w2,]
  y_ij1 <- x[,0:(h - 2L), 1:w1,]
  
  a <- tf$square(y_ij - y_i1j)
  b <- tf$square(y_ij - y_ij1)
  tf$reduce_sum(tf$pow(a + b, 1.25))
}

# main <- reticulate::py_run_string("
# img_height = 600
# img_width = 600
# from keras import backend as K
# def continuity_loss(x):
#   # continuity loss util function
#   assert K.ndim(x) == 4
#   if K.image_data_format() == 'channels_first':
#     a = K.square(x[:, :, :img_height - 1, :img_width - 1] -
#     x[:, :, 1:, :img_width - 1])
#     b = K.square(x[:, :, :img_height - 1, :img_width - 1] -
#     x[:, :, :img_height - 1, 1:])
#   else:
#     a = K.square(x[:, :img_height - 1, :img_width - 1, :] -
#     x[:, 1:, :img_width - 1, :])
#     b = K.square(x[:, :img_height - 1, :img_width - 1, :] -
#     x[:, :img_height - 1, 1:, :])
#   return K.sum(K.pow(a + b, 1.25))
# ")
# main$continuity_loss(tf$constant(img))
# total_variation_loss(img, 600, 600)

# Data Preparation --------------------------------------------------------

# some settings we found interesting
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

# the settings we will use in this experiment
settings <- saved_settings$dreamy


# Model definition --------------------------------------------------------

img_height <- 600
img_width <- 600
img_size <- c(img_height, img_width, 3)

# this will contain our generated image
dream <- layer_input(batch_shape = c(1, img_size))

# build the VGG16 network with our placeholder
# the model will be loaded with pre-trained ImageNet weights
model <- tf$contrib$keras$applications$vgg16$VGG16(
  input_tensor = dream,
  weights = "imagenet",  
  include_top=FALSE)

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict <- model$layers
names(layer_dict) <- map_chr(layer_dict ,~.x$name)

# define the loss
loss <- tf$Variable(0.0)
for(layer_name in names(settings$features)){
  # add the L2 norm of the features of a layer to the loss
  stopifnot(layer_name %in% names(layer_dict))
  coeff <- settings$features[[layer_name]]
  x <- layer_dict[[layer_name]]$output
  out_shape <- layer_dict[[layer_name]]$output_shape %>% unlist()
  # we avoid border artifacts by only involving non-border pixels in the loss
  loss <- loss + coeff*tf$reduce_sum(
    tf$square(x[,3:(out_shape[2] - 1), 3:(out_shape[2] - 1),])
  )/prod(out_shape[-1])
}

# add continuity loss (gives image local coherence, can result in an artful blur)
loss <- loss + settings$continuity*
  total_variation_loss(x = dream, 600L, 600L)/
  prod(img_size)
# add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
loss <- loss + settings$dream_l2*tf$reduce_sum(tf$square(dream))/prod(img_size)



# image_path <- "david-bowie.jpg"
# height = 600
# width = 600






