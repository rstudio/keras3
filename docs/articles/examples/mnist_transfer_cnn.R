
library(keras)

now <- Sys.time()

batch_size <- 128
num_classes <- 5
epochs <- 5

# input image dimensions
img_rows <- 28
img_cols <- 28

# number of convolutional filters to use
filters <- 32

# size of pooling area for max pooling
pool_size <- 2

# convolution kernel size
kernel_size <- 3

# input shape
input_shape <- c(img_rows, img_cols, 1)


