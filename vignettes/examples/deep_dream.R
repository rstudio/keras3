

# Setup

library(keras)
library(tensorflow)


base_image_path = get_file('paris.jpg', 'https://i.imgur.com/aGBdQyK.jpg')
result_prefix = 'sky_dream'

# These are the names of the layers
# for which we try to maximize activation,
# as well as their weight in the final loss
# we try to maximize.
# You can tweak these setting to obtain new visual effects.
layer_settings = list(
  'mixed4' = 1.0,
  'mixed5' = 1.5,
  'mixed6' = 2.0,
  'mixed7' = 2.5
)

# Playing with these hyperparameters will also allow you to achieve new effects
step = 0.01  # Gradient ascent step size
num_octave = 3  # Number of scales at which to run gradient ascent
octave_scale = 1.4  # Size ratio between scales
iterations = 20  # Number of ascent steps per scale
max_loss = 15.

# This is our base image:
plot(magick::image_read(base_image_path))

# Let's set up some image preprocessing/deprocessing utilities:
preprocess_image <- function(image_path) {
  # Util function to open, resize and format pictures
  # into appropriate arrays.
  img = tf$keras$preprocessing$image$load_img(image_path)
  img = tf$keras$preprocessing$image$img_to_array(img)
  img = tf$expand_dims(img, axis=0L)
  img = inception_v3_preprocess_input(img)
  img
}


deprocess_image <- function(x) {
  x = array_reshape(x, dim = c(dim(img)[[2]], dim(img)[[3]], 3))
  # Undo inception v3 preprocession
  x = x / 2.
  x =  x + 0.5
  x = x * 255.
  # Convert to uint8 and clip to the valid range [0, 255]
  x = tf$clip_by_value(x, 0L, 255L) %>% tf$cast(dtype = 'uint8')
  x
}

save_img <- function(img, fname) {
  img <- deprocess_image(img)
  image_array_save(img, fname)
}



# Build an InceptionV3 model loaded with pre-trained ImageNet weights
model <- application_inception_v3(weights = "imagenet",
                                  include_top = FALSE)

# Get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = list()
for (layer_name in names(layer_settings)) {
  coeff <- layer_settings[[layer_name]]
  # Retrieves the layer's output
  activation <- get_layer(model, layer_name)$output
  outputs_dict[[layer_name]] <- activation
}


# Set up a model that returns the activation values for every target layer
# (as a named list)
feature_extractor = keras_model(inputs = model$inputs,
                                outputs = outputs_dict)

compute_loss <- function(input_image) {
  features = feature_extractor(input_image)
  names(features) = names(layer_settings)
  loss = tf$zeros(shape=list())
  for (names in names(layer_settings)) {
    coeff = layer_settings[[names]]
    activation = features[[names]]
    # We avoid border artifacts by only involving non-border pixels in the loss.
    scaling = tf$reduce_prod(tf$cast(tf$shape(activation), 'float32'))
    loss = loss + coeff * tf$reduce_sum(tf$square(activation)) / scaling
  }
  loss
}

# Set up the gradient ascent loop for one octave
gradient_ascent_step <- function(img, learning_rate) {
  with(tf$GradientTape() %as% tape, {
    tape$watch(img)
    loss = compute_loss(img)
  })
  # Compute gradients.
  grads = tape$gradient(loss, img)
  # Normalize gradients.
  grads = grads / tf$maximum(tf$reduce_mean(tf$abs(grads)), 1e-6)
  img = img + learning_rate * grads
  list(loss, img)
}

gradient_ascent_loop <- function(img, iterations, learning_rate, max_loss = NULL) {
  for (i in 1:iterations) {
    c(loss, img) %<-% gradient_ascent_step(img, learning_rate)
    if (!is.null(max_loss) && as.array(loss) > max_loss)
      break
    cat("...Loss value at step", i, ":", as.array(loss), "\n")
  }
  img
}

# Run the training loop, iterating over different octaves
original_img = preprocess_image(base_image_path)

# Prepares a list of shape tuples defining the different scales at which to run gradient ascent
original_shape <- dim(original_img)[2:3]

successive_shapes <- list(original_shape)

for (i in 1:num_octave) {
  shape <- as.integer(original_shape / (octave_scale ^ i))
  successive_shapes[[length(successive_shapes) + 1]] <- shape
}

# Reverses the list of shapes so they're in increasing order
successive_shapes <- rev(successive_shapes[1:3])
#  Resizes the array of the image to the smallest scale
shrunk_original_img <- tf$image$resize(original_img, successive_shapes[[1]])

img = tf$identity(original_img)  # Make a copy

for (i in 1:length(successive_shapes)) {
  shape = successive_shapes[[i]]
  cat("Processing octave", i, "with shape", shape, "\n")
  # Scales up the dream image
  img <- tf$image$resize(img, shape)
  # Runs gradient ascent, altering the dream
  img <- gradient_ascent_loop(img,
                              iterations = iterations,
                              learning_rate = step,
                              max_loss = max_loss)
  # Scales up the smaller version of the original image: it will be pixellated
  upscaled_shrunk_original_img <-
    tf$image$resize(shrunk_original_img, shape)
  # Computes the high-quality version of the original image at this size
  same_size_original <-
    tf$image$resize(original_img, shape)
  # The difference between the two is the detail that was lost when scaling up
  lost_detail <-
    same_size_original - upscaled_shrunk_original_img
  # Reinjects lost detail into the dream
  img <- img + lost_detail
  shrunk_original_img <-
    tf$image$resize(original_img, shape)
  tf$keras$preprocessing$image$save_img(paste(result_prefix,'.png',sep = ''), deprocess_image(img$numpy()))
}

# Plot result
plot(magick::image_read(paste(result_prefix,'.png',sep = '')))
