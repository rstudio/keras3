#' Neural style transfer with Keras.
#' 
#' It is preferable to run this script on a GPU, for speed.
#' 
#' Example result: https://twitter.com/fchollet/status/686631033085677568
#' 
#' Style transfer consists in generating an image
#' with the same "content" as a base image, but with the
#' "style" of a different picture (typically artistic). 
#' 
#' This is achieved through the optimization of a loss function
#' that has 3 components: "style loss", "content loss",
#' and "total variation loss":
#' 
#'  - The total variation loss imposes local spatial continuity between
#'    the pixels of the combination image, giving it visual coherence.
#' 
#'  - The style loss is where the deep learning keeps in --that one is defined
#'    using a deep convolutional neural network. Precisely, it consists in a sum of
#'    L2 distances between the Gram matrices of the representations of
#'    the base image and the style reference image, extracted from
#'    different layers of a convnet (trained on ImageNet). The general idea
#'    is to capture color/texture information at different spatial
#'    scales (fairly large scales --defined by the depth of the layer considered).
#' 
#'  - The content loss is a L2 distance between the features of the base
#'    image (extracted from a deep layer) and the features of the combination image,
#'    keeping the generated image close enough to the original one.
#'
library(keras)
library(purrr)
library(R6)

# Parameters --------------------------------------------------------------

base_image_path <- "neural-style-base-img.png"
style_reference_image_path <- "neural-style-style.jpg"
iterations <- 10

# these are the weights of the different loss components
total_variation_weight <- 1
style_weight <- 1
content_weight <- 0.025

# dimensions of the generated picture.
img <- image_load(base_image_path)
width <- img$size[[1]]
height <- img$size[[2]]
img_nrows <- 400
img_ncols <- as.integer(width * img_nrows / height)


# Functions ---------------------------------------------------------------

# util function to open, resize and format pictures into appropriate tensors
preprocess_image <- function(path){
  img <- image_load(path, target_size = c(img_nrows, img_ncols)) %>%
    image_to_array() %>%
    array_reshape(c(1, dim(.)))
  imagenet_preprocess_input(img)
}

# util function to convert a tensor into a valid image
# also turn BGR into RGB.
deprocess_image <- function(x){
  x <- x[1,,,]
  # Remove zero-center by mean pixel
  x[,,1] <- x[,,1] + 103.939
  x[,,2] <- x[,,2] + 116.779
  x[,,3] <- x[,,3] + 123.68
  # BGR -> RGB
  x <- x[,,c(3,2,1)]
  # clip to interval 0, 255
  x[x > 255] <- 255
  x[x < 0] <- 0
  x[] <- as.integer(x)/255
  x
}


# Defining the model ------------------------------------------------------

# get tensor representations of our images
base_image <- k_variable(preprocess_image(base_image_path))
style_reference_image <- k_variable(preprocess_image(style_reference_image_path))

# this will contain our generated image
combination_image <- k_placeholder(c(1, img_nrows, img_ncols, 3))

# combine the 3 images into a single Keras tensor
input_tensor <- k_concatenate(list(base_image, style_reference_image, 
                                   combination_image), axis = 1)

# build the VGG16 network with our 3 images as input
# the model will be loaded with pre-trained ImageNet weights
model <- application_vgg16(input_tensor = input_tensor, weights = "imagenet", 
                           include_top = FALSE)

print("Model loaded.")

nms <- map_chr(model$layers, ~.x$name)
output_dict <- map(model$layers, ~.x$output) %>% set_names(nms)

# compute the neural style loss
# first we need to define 4 util functions

# the gram matrix of an image tensor (feature-wise outer product)

gram_matrix <- function(x){
  
  features <- x %>%
    k_permute_dimensions(pattern = c(3, 1, 2)) %>%
    k_batch_flatten()
  
  k_dot(features, k_transpose(features))
}

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image

style_loss <- function(style, combination){
  S <- gram_matrix(style)
  C <- gram_matrix(combination)
  
  channels <- 3
  size <- img_nrows*img_ncols
  
  k_sum(k_square(S - C)) / (4 * channels^2  * size^2)
}

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image

content_loss <- function(base, combination){
  k_sum(k_square(combination - base))
}

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent

total_variation_loss <- function(x){
  y_ij  <- x[,1:(img_nrows - 1L), 1:(img_ncols - 1L),]
  y_i1j <- x[,2:(img_nrows), 1:(img_ncols - 1L),]
  y_ij1 <- x[,1:(img_nrows - 1L), 2:(img_ncols),]
  
  a <- k_square(y_ij - y_i1j)
  b <- k_square(y_ij - y_ij1)
  k_sum(k_pow(a + b, 1.25))
}

# combine these loss functions into a single scalar
loss <- k_variable(0.0)
layer_features <- output_dict$block4_conv2
base_image_features <- layer_features[1,,,]
combination_features <- layer_features[3,,,]

loss <- loss + content_weight*content_loss(base_image_features, 
                                           combination_features)

feature_layers = c('block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1')

for(layer_name in feature_layers){
  layer_features <- output_dict[[layer_name]]
  style_reference_features <- layer_features[2,,,]
  combination_features <- layer_features[3,,,]
  sl <- style_loss(style_reference_features, combination_features)
  loss <- loss + ((style_weight / length(feature_layers)) * sl)
}

loss <- loss + (total_variation_weight * total_variation_loss(combination_image))

# get the gradients of the generated image wrt the loss
grads <- k_gradients(loss, combination_image)[[1]]

f_outputs <- k_function(list(combination_image), list(loss, grads))

eval_loss_and_grads <- function(image){
  image <- array_reshape(image, c(1, img_nrows, img_ncols, 3))
  outs <- f_outputs(list(image))
  list(
    loss_value = outs[[1]],
    grad_values = array_reshape(outs[[2]], dim = length(outs[[2]]))
  )
}

# Loss and gradients evaluator.
# 
# This Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.
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

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
dms <- c(1, img_nrows, img_ncols, 3)
x <- array(data = runif(prod(dms), min = 0, max = 255) - 128, dim = dms)

# Run optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the loss
for(i in 1:iterations){

  # Run L-BFGS
  opt <- optim(
    array_reshape(x, dim = length(x)), fn = evaluator$loss, gr = evaluator$grads, 
    method = "L-BFGS-B",
    control = list(maxit = 15)
  )
  
  # Print loss value
  print(opt$value)
  
  # decode the image
  image <- x <- opt$par
  image <- array_reshape(image, dms)
  
  # plot
  im <- deprocess_image(image)
  plot(as.raster(im))
}

