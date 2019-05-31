#' In this example we fine tune Mobile Net to better predict cats and
#' dogs in photos. It also demonstrates the usage of image data generators
#' for efficient preprocessing and training.
#' 
#' It's preferable to run this example in a GPU.

# Download data -----------------------------------------------------------

download.file(
  "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip", 
  destfile = "cats-dogs.zip"
)

# Pre-processing ----------------------------------------------------------

zip::unzip("cats-dogs.zip", exdir = "data-raw")

# We will organize images in the following structure:
# data/
#     train/
#          Cat/
#          Dog/
#     validation
#          Cat/
#          Dog/
#     test/
#          images/
#

all_imgs <- fs::dir_ls(
  "data-raw/PetImages/", 
  recursive = TRUE, 
  type = "file",
  glob = "*.jpg"
)

# some images are corrupt and we exclude them
# this will make sure all images can be read.
for (im in all_imgs) {
  out <- try(magick::image_read(im), silent = TRUE)
  if (inherits(out, "try-error")) {
    fs::file_delete(im)
    message("removed image: ", im)
  }
}

# re-list all imgs
all_imgs <- fs::dir_ls(
  "data-raw/PetImages/", 
  recursive = TRUE, 
  type = "file",
  glob = "*.jpg"
)

set.seed(5)

training_imgs <- sample(all_imgs, size = length(all_imgs)/2)
validation_imgs <- sample(all_imgs[!all_imgs %in% training_imgs], size = length(all_imgs)/4)         
testing_imgs <- all_imgs[!all_imgs %in% c(training_imgs, validation_imgs)]

# create directory structure
fs::dir_create(c(
  "data/train/Cat",
  "data/train/Dog",
  "data/validation/Cat",
  "data/validation/Dog",
  "data/test/images"
))

# copy training images
fs::file_copy(
  path = training_imgs, 
  new_path = gsub("data-raw/PetImages", "data/train", training_imgs)
)

# copy valid images
fs::file_copy(
  path = validation_imgs, 
  new_path = gsub("data-raw/PetImages", "data/validation", validation_imgs)
)

# copy testing imgs
fs::file_copy(
  path = testing_imgs,
  new_path = gsub("data-raw/PetImages/(Dog|Cat)/", "data/test/images/\\1", testing_imgs)
)

# Image flow --------------------------------------------------------------

library(keras)

training_image_gen <- image_data_generator(
  rotation_range = 20,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  horizontal_flip = TRUE,
  preprocessing_function = imagenet_preprocess_input
)

validation_image_gen <- image_data_generator(
  preprocessing_function = imagenet_preprocess_input
)

training_image_flow <- flow_images_from_directory(
  directory = "data/train/", 
  generator = training_image_gen, 
  class_mode = "binary",
  batch_size = 100,
  target_size = c(224, 224), 
)

validation_image_flow <- flow_images_from_directory(
  directory = "data/validation/", 
  generator = validation_image_gen, 
  class_mode = "binary",
  batch_size = 100,
  target_size = c(224, 224), 
  shuffle = FALSE
)

# Model -------------------------------------------------------------------

mob <- application_mobilenet(include_top = FALSE, pooling = "avg")
freeze_weights(mob)

model <- keras_model_sequential() %>% 
  mob() %>% 
  layer_dense(256, activation = "relu") %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% 
  compile(loss = "binary_crossentropy", optimizer = "adam", metrics = "accuracy")

model %>% fit_generator(
  generator = training_image_flow, 
  epochs = 1, 
  steps_per_epoch = training_image_flow$n/training_image_flow$batch_size,
  validation_data = validation_image_flow,
  validation_steps = validation_image_flow$n/validation_image_flow$batch_size
)

# now top layers weights are fine, we can unfreeze the lower layer weights.
unfreeze_weights(mob)

model %>% 
  compile(loss = "binary_crossentropy", optimizer = "adam", metrics = "accuracy")

model %>% fit_generator(
  generator = training_image_flow, 
  epochs = 3, 
  steps_per_epoch = training_image_flow$n/training_image_flow$batch_size,
  validation_data = validation_image_flow,
  validation_steps = validation_image_flow$n/validation_image_flow$batch_size
)

# Generate predictions for test data --------------------------------------

test_flow <- flow_images_from_directory(
  generator = validation_image_gen,
  directory = "data/test", 
  target_size = c(224, 224),
  class_mode = NULL,
  shuffle = FALSE
)

predictions <- predict_generator(
  model, 
  test_flow,
  steps = test_flow$n/test_flow$batch_size
)

magick::image_read(testing_imgs[1])
predictions[1]

magick::image_read(testing_imgs[6250])
predictions[6250]

