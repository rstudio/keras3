#' To run this example:
#' 
#'  1) Download the train.zip and train_masks.zip files from: 
#'     https://www.kaggle.com/c/carvana-image-masking-challenge/data
#' 
#'  2) Create an "input" directory and extract the zip files into it (after this there
#'     should be "train" and "train_masks" subdirectories within the "input" directory).
#'

#` This code runs only on Linux because of specific parallel backend. 
#` You can find Windows version here: https://keras.rstudio.com/articles/examples/unet.html
#` unet architecture is based on original Python code 
#` from https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge. 
#` It shows an example of creating custom architectures in R version of keras 
#` and working with images using magick package. 
#` parallel + doParallel + foreach allows to speed up the code.
#` You can download the data from https://www.kaggle.com/c/carvana-image-masking-challenge

library(keras)
library(magick)
library(abind)
library(reticulate)
library(doMC)
library(foreach)


# Parameters -----------------------------------------------------

input_size <- 128

epochs <- 30
batch_size <- 16

orig_width <- 1918
orig_height <- 1280

threshold <- 0.5

train_samples <- 5088
train_index <- sample(1:train_samples, round(train_samples * 0.8)) # 80%
val_index <- c(1:train_samples)[-train_index]

images_dir <- "./input/train/" 
masks_dir <- "./input/train_masks/"


# Loss function -----------------------------------------------------

K <- backend()

dice_coef <- function(y_true, y_pred, smooth = 1.0) {
	y_true_f <- k_flatten(y_true)
	y_pred_f <- k_flatten(y_pred)
	intersection <- k_sum(y_true_f * y_pred_f)
	result <- (2 * intersection + smooth) / 
		(k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
	return(result)
}

bce_dice_loss <- function(y_true, y_pred) {
	result <- loss_binary_crossentropy(y_true, y_pred) +
		(1 - dice_coef(y_true, y_pred))
	return(result)
}


# U-net 128 -----------------------------------------------------

get_unet_128 <- function(input_shape = c(128, 128, 3),
						 num_classes = 1) {
	
	inputs <- layer_input(shape = input_shape)
	# 128
	
	down1 <- inputs %>%
		layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") %>%
		layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") 
	down1_pool <- down1 %>%
		layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
	# 64
	
	down2 <- down1_pool %>%
		layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") %>%
		layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") 
	down2_pool <- down2 %>%
		layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
	# 32
	
	down3 <- down2_pool %>%
		layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") %>%
		layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") 
	down3_pool <- down3 %>%
		layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
	# 16
	
	down4 <- down3_pool %>%
		layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") %>%
		layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") 
	down4_pool <- down4 %>%
		layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
	# 8
	
	center <- down4_pool %>%
		layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") %>%
		layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") 
	# center
	
	up4 <- center %>%
		layer_upsampling_2d(size = c(2, 2)) %>%
		{layer_concatenate(inputs = list(down4, .), axis = 3)} %>%
		layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") %>%
		layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") %>%
		layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu")
	# 16
	
	up3 <- up4 %>%
		layer_upsampling_2d(size = c(2, 2)) %>%
		{layer_concatenate(inputs = list(down3, .), axis = 3)} %>%
		layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") %>%
		layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") %>%
		layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu")
	# 32
	
	up2 <- up3 %>%
		layer_upsampling_2d(size = c(2, 2)) %>%
		{layer_concatenate(inputs = list(down2, .), axis = 3)} %>%
		layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") %>%
		layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") %>%
		layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu")
	# 64
	
	up1 <- up2 %>%
		layer_upsampling_2d(size = c(2, 2)) %>%
		{layer_concatenate(inputs = list(down1, .), axis = 3)} %>%
		layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") %>%
		layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu") %>%
		layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
		layer_batch_normalization() %>%
		layer_activation("relu")
	# 128
	
	classify <- layer_conv_2d(up1,
							  filters = num_classes, 
							  kernel_size = c(1, 1),
							  activation = "sigmoid")
	
	
	model <- keras_model(
		inputs = inputs,
		outputs = classify
	)
	
	model %>% compile(
		optimizer = optimizer_rmsprop(lr = 0.0001),
		loss = bce_dice_loss,
		metrics = custom_metric("dice_coef", dice_coef)
	)
	
	return(model)
}

model <- get_unet_128()


# Read and augmentation functions -----------------------------------------------------

imagesRead <- function(image_file,
					   mask_file,
					   target_width = 128, 
					   target_height = 128) {
	img <- image_read(image_file)
	img <- image_scale(img, paste0(target_width, "x", target_height, "!"))
	
	mask <- image_read(mask_file)
	mask <- image_scale(mask, paste0(target_width, "x", target_height, "!"))
	return(list(img = img, mask = mask))
}

randomBSH <- function(img,
					  u = 0,
					  brightness_shift_lim = c(90, 110), # percentage
					  saturation_shift_lim = c(95, 105), # of current value
					  hue_shift_lim = c(80, 120)) {
	
	if (rnorm(1) < u) return(img)
	
	brightness_shift <- runif(1, 
							  brightness_shift_lim[1], 
							  brightness_shift_lim[2])
	saturation_shift <- runif(1, 
							  saturation_shift_lim[1], 
							  saturation_shift_lim[2])
	hue_shift <- runif(1, 
					   hue_shift_lim[1], 
					   hue_shift_lim[2])
	
	img <- image_modulate(img, 
						  brightness = brightness_shift, 
						  saturation =  saturation_shift, 
						  hue = hue_shift)
	img
}

img2arr <- function(image, 
					target_width = 128,
					target_height = 128) {
	result <- aperm(as.numeric(image[[1]])[, , 1:3], c(2, 1, 3)) # transpose
	array_reshape(result, c(1, target_width, target_height, 3))
}

mask2arr <- function(mask,
					 target_width = 128,
					 target_height = 128) {
	result <- t(as.numeric(mask[[1]])[, , 1]) # transpose
	array_reshape(result, c(1, target_width, target_height, 1))
}


# Iterators with parallel processing -----------------------------------------------------

registerDoMC(4)

train_generator <- function(images_dir, 
							samples_index,
							masks_dir, 
							batch_size) {
	images_iter <- list.files(images_dir, 
							  pattern = ".jpg", 
							  full.names = TRUE)[samples_index] # for current epoch
	images_all <- list.files(images_dir, 
							 pattern = ".jpg",
							 full.names = TRUE)[samples_index]  # for next epoch
	masks_iter <- list.files(masks_dir, 
							 pattern = ".gif",
							 full.names = TRUE)[samples_index] # for current epoch
	masks_all <- list.files(masks_dir, 
							pattern = ".gif",
							full.names = TRUE)[samples_index] # for next epoch
	
	function() {
		
		# start new epoch
		if (length(images_iter) < batch_size) {
			images_iter <<- images_all
			masks_iter <<- masks_all
		}
		
		batch_ind <- sample(1:length(images_iter), batch_size)
		
		batch_images_list <- images_iter[batch_ind]
		images_iter <<- images_iter[-batch_ind]
		batch_masks_list <- masks_iter[batch_ind]
		masks_iter <<- masks_iter[-batch_ind]
		
		
		x_y_batch <- foreach(i = 1:batch_size) %dopar% {
			x_y_imgs <- imagesRead(image_file = batch_images_list[i],
								   mask_file = batch_masks_list[i])
			# augmentation
			x_y_imgs$img <- randomBSH(x_y_imgs$img)
			# return as arrays
			x_y_arr <- list(x = img2arr(x_y_imgs$img),
							y = mask2arr(x_y_imgs$mask))
		}
		
		x_y_batch <- purrr::transpose(x_y_batch)
		
		x_batch <- do.call(abind, c(x_y_batch$x, list(along = 1)))
		
		y_batch <- do.call(abind, c(x_y_batch$y, list(along = 1)))
		
		result <- list(keras_array(x_batch), keras_array(y_batch))
		return(result)
	}
}

val_generator <- function(images_dir, 
						  samples_index,
						  masks_dir, 
						  batch_size) {
	images_iter <- list.files(images_dir, 
							  pattern = ".jpg", 
							  full.names = TRUE)[samples_index] # for current epoch
	images_all <- list.files(images_dir, 
							 pattern = ".jpg",
							 full.names = TRUE)[samples_index]  # for next epoch
	masks_iter <- list.files(masks_dir, 
							 pattern = ".gif",
							 full.names = TRUE)[samples_index] # for current epoch
	masks_all <- list.files(masks_dir, 
							pattern = ".gif",
							full.names = TRUE)[samples_index] # for next epoch
	
	function() {
		
		# start new epoch
		if (length(images_iter) < batch_size) {
			images_iter <<- images_all
			masks_iter <<- masks_all
		}
		
		batch_ind <- sample(1:length(images_iter), batch_size)
		
		batch_images_list <- images_iter[batch_ind]
		images_iter <<- images_iter[-batch_ind]
		batch_masks_list <- masks_iter[batch_ind]
		masks_iter <<- masks_iter[-batch_ind]
		
		
		x_y_batch <- foreach(i = 1:batch_size) %dopar% {
			x_y_imgs <- imagesRead(image_file = batch_images_list[i],
								   mask_file = batch_masks_list[i])
			# without augmentation
			
			# return as arrays
			x_y_arr <- list(x = img2arr(x_y_imgs$img),
							y = mask2arr(x_y_imgs$mask))
		}
		
		x_y_batch <- purrr::transpose(x_y_batch)
		
		x_batch <- do.call(abind, c(x_y_batch$x, list(along = 1)))
		
		y_batch <- do.call(abind, c(x_y_batch$y, list(along = 1)))
		
		result <- list(keras_array(x_batch), keras_array(y_batch))
		return(result)
	}
}

train_iterator <- py_iterator(train_generator(images_dir = images_dir,
											  masks_dir = masks_dir,
											  samples_index = train_index,
											  batch_size = batch_size))

val_iterator <- py_iterator(val_generator(images_dir = images_dir,
										  masks_dir = masks_dir,
										  samples_index = val_index,
										  batch_size = batch_size))


# Training -----------------------------------------------------

tensorboard("logs_r")

callbacks_list <- list(
	callback_tensorboard("logs_r"),
	callback_early_stopping(monitor = "val_python_function",
							min_delta = 1e-4,
							patience = 8,
							verbose = 1,
							mode = "max"),
	callback_reduce_lr_on_plateau(monitor = "val_python_function",
								  factor = 0.1,
								  patience = 4,
								  verbose = 1,
								  epsilon = 1e-4,
								  mode = "max"),
	callback_model_checkpoint(filepath = "weights_r/unet128_{epoch:02d}.h5",
							  monitor = "val_python_function",
							  save_best_only = TRUE,
							  save_weights_only = TRUE, 
							  mode = "max" )
)

model %>% fit_generator(
	train_iterator,
	steps_per_epoch = as.integer(length(train_index) / batch_size), 
	epochs = epochs, 
	validation_data = val_iterator,
	validation_steps = as.integer(length(val_index) / batch_size),
	verbose = 1,
	callbacks = callbacks_list
)

