
library(keras)
use_virtualenv("~/tensorflow")

# generate dummy training data
data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)

# genereate dummy input data
input <- matrix(rexp(10*784), nrow = 10, ncol = 784)

# define and compile the model
model <- keras_model_sequential() %>% 
  layer_dense(32, input_shape = 784) %>% 
  layer_activation(activation_relu) %>% 
  layer_dense(10) %>% 
  layer_activation('softmax') %>% 
  compile( 
    loss= 'binary_crossentropy', 
    optimizer = optimizer_sgd(),
    metrics = metric_binary_accuracy()
  )

# train the model 
model <- fit(model, data, labels)

# make some predictions
predict(model, input)

# save the model and load it back in
save_model_hdf5(model, "model.hdf5")
model <- load_model_hdf5("model.hdf5")




