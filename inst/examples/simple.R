

library(keras)

# Training and input data -------------------------------------------

# generate dummy training data
data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)

# genereate dummy input data
input <- matrix(rexp(10*784), nrow = 10, ncol = 784)


# Sequential API ----------------------------------------------------------

# define and train the model
model <- sequential_model() %>% 
  layer_dense(32, input_dim = 784) %>% 
  layer_activation('relu') %>% 
  layer_dense(10) %>% 
  layer_activation('softmax') %>% 
  compile(loss='binary_crossentropy',
          optimizer='rmsprop',
          metrics='accuracy') %>%
  fit(data, labels)


# make some predictions
model %>% 
  predict(input)


# Functional API ----------------------------------------------------------

# define input tensor
inputs <- layer_input(784) 

# define prediction layers
predictions <- inputs %>% 
  layer_dense(64, activation = 'relu') %>% 
  layer_dense(64, activation = 'relu') %>% 
  layer_dense(10, activation = 'softmax')

# define and train the model
model <- model(input = inputs, output = predictions) %>% 
  compile(loss='binary_crossentropy',
          optimizer='rmsprop',
          metrics='accuracy') %>% 
  fit(data, labels)

# make some predictions
model %>% 
  predict(input)
  
