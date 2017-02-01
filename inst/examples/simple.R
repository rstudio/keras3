

library(keras)
library(magrittr)

model <- sequential_model() %>% 
  layer_dense(output_dim = 32L, input_dim = 784L) %>% 
  layer_activation('relu') %>% 
  layer_dense(output_dim = 10L) %>% 
  layer_activation('softmax')

print(model$layers)