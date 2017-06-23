# This script demonstrates how to build a variational autoencoder with Keras.
# Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
library(keras)
K <- keras::backend()

# Parameters --------------------------------------------------------------

batch_size <- 100L
original_dim <- 784L
latent_dim <- 2L
intermediate_dim <- 256L
epochs <- 50L
epsilon_std <- 1.0

# Model definition --------------------------------------------------------

x <- layer_input(batch_shape = c(batch_size, original_dim))
h <- layer_dense(x, intermediate_dim, activation = "relu")
z_mean <- layer_dense(h, latent_dim)
z_log_var <- layer_dense(h, latent_dim)

sampling <- function(arg){
  z_mean <- arg[,0:1]
  z_log_var <- arg[,2:3]
  
  epsilon <- K$random_normal(
    shape = c(batch_size, latent_dim), 
    mean=0.,
    stddev=epsilon_std
  )
  
  z_mean + K$exp(z_log_var/2)*epsilon
}

# note that "output_shape" isn't necessary with the TensorFlow backend
z <- layer_concatenate(list(z_mean, z_log_var)) %>% 
  layer_lambda(sampling)

# we instantiate these layers separately so as to reuse them later
decoder_h <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
h_decoded <- decoder_h(z)
x_decoded_mean <- decoder_mean(h_decoded)

# end-to-end autoencoder
vae <- keras_model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder <- keras_model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input <- layer_input(shape = latent_dim)
h_decoded_2 <- decoder_h(decoder_input)
x_decoded_mean_2 <- decoder_mean(h_decoded_2)
generator <- keras_model(decoder_input, x_decoded_mean_2)


vae_loss <- function(x, x_decoded_mean){
  xent_loss <- (original_dim/1.0)*loss_binary_crossentropy(x, x_decoded_mean)
  kl_loss <- -0.5*K$mean(1 + z_log_var - K$square(z_mean) - K$exp(z_log_var), axis = -1L)
  xent_loss + kl_loss
}

vae %>% compile(optimizer = "rmsprop", loss = vae_loss)


# Data preparation --------------------------------------------------------

mnist <- dataset_mnist()
x_train <- mnist$train$x/255
x_test <- mnist$test$x/255
x_train <- x_train %>% apply(1, as.numeric) %>% t()
x_test <- x_test %>% apply(1, as.numeric) %>% t()


# Model training ----------------------------------------------------------

vae %>% fit(
  x_train, x_train, 
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size, 
  validation_data = list(x_test, x_test)
)


# Visualizations ----------------------------------------------------------

library(ggplot2)
library(dplyr)
x_test_encoded <- predict(encoder, x_test, batch_size = batch_size)

x_test_encoded %>%
  as_data_frame() %>% 
  mutate(class = as.factor(mnist$test$y)) %>%
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()

# display a 2D manifold of the digits
n <- 15  # figure with 15x15 digits
digit_size <- 28

# we will sample n points within [-15, 15] standard deviations
grid_x <- seq(-15, 15, length.out = n)
grid_y <- seq(-15, 15, length.out = n)

for(i in 1:length(grid_x)){
  for(j in 1:length(grid_y)){
    
  }
}

# figure = np.zeros((digit_size * n, digit_size * n))
# # we will sample n points within [-15, 15] standard deviations
# grid_x = np.linspace(-15, 15, n)
# grid_y = np.linspace(-15, 15, n)
# 
# for i, yi in enumerate(grid_x):
#   for j, xi in enumerate(grid_y):
#   z_sample = np.array([[xi, yi]]) * epsilon_std
# x_decoded = generator.predict(z_sample)
# digit = x_decoded[0].reshape(digit_size, digit_size)
# figure[i * digit_size: (i + 1) * digit_size,
#        j * digit_size: (j + 1) * digit_size] = digit
# 
# plt.figure(figsize=(10, 10))
# plt.imshow(figure)
# plt.show()

  


