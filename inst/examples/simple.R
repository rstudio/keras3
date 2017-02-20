
library(keras)
use_virtualenv("~/tensorflow")

# generate dummy training data
data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)

# genereate dummy input data
input <- matrix(rexp(10*784), nrow = 10, ncol = 784)

# define and compile the model
model <- model_sequential() %>% 
  layer_dense(32, input_dim = 784) %>% 
  layer_activation('relu') %>% 
  layer_dense(10) %>% 
  layer_activation('softmax') %>% 
  compile( 
    loss='binary_crossentropy', 
    optimizer = optimizer_sgd(),
    metrics='accuracy'
  )

# train the model 
model <- fit(model, data, labels)

# make some predictions
predict(model, input)

# save the model and load it back in
write_model(model, "model.hdf5")
model <- read_model("model.hdf5")




