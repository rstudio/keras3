

library(keras)
use_virtualenv("~/tensorflow")


# Training and input data -------------------------------------------

# generate dummy training data
data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)

# genereate dummy input data
input <- matrix(rexp(10*784), nrow = 10, ncol = 784)


# Sequential API ----------------------------------------------------------

# define and compile the model (these methods copy the model prior to
# returning it so do not mutate the model)
model <- model_sequential() %>% 
  layer_dense(32, input_dim = 784) %>% 
  layer_activation('relu') %>% 
  layer_dense(10) %>% 
  layer_activation('softmax') %>% 
  compile(loss='binary_crossentropy',
          optimizer = optimizer_sgd(),
          metrics='accuracy')

# save the model and load it back in
model %>% save_model("model.hdf5")
model <- load_model("model.hdf5")

# train the model (this definitely mutates the model object)
fit(model, data, labels)

# make some predictions (this could in theory mutate the model
# object however it probably doesn't)
predict(model, input)


# Functional API ----------------------------------------------------------

# define input tensor
inputs <- layer_input(784) 

# define prediction layers
predictions <- inputs %>% 
  layer_dense(64, activation = 'relu') %>% 
  layer_dense(64, activation = 'relu') %>% 
  layer_dense(10, activation = 'softmax')

# define the model
model <- model(input = inputs, output = predictions) %>% 
  compile(loss='binary_crossentropy',
          optimizer = optimizer_rmsprop(),
          metrics='accuracy')
  
# train the model
model %>% fit(data, labels)

# make some predictions
model %>% predict(input)
  
