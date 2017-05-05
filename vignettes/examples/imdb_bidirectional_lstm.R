#' Train a Bidirectional LSTM on the IMDB sentiment classification task.
#' 
#' Output after 4 epochs on CPU: ~0.8146
#' Time per epoch on CPU (Core i7): ~150s.

library(keras)

max_features <- 20000

# cut texts after this number of words
# (among top max_features most common words)
maxlen <- 100

batch_size <- 32

cat('Loading data...\n')
imdb <- dataset_imdb(num_words = max_features)
x_train <- imdb$train$x
y_train <- imdb$train$y
x_test <- imdb$test$x
y_test <- imdb$test$y

cat(length(x_train), 'train sequences\n')
cat(length(x_test), 'test sequences\n')

cat('Pad sequences (samples x time)\n')
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)
cat('x_train shape:', dim(x_train), '\n')
cat('x_test shape:', dim(x_test), '\n')

model <- keras_model_sequential()
model %>%
  layer_embedding(input_dim = max_features, output_dim = 128, input_length = maxlen) %>% 
  bidirectional(layer_lstm(units = 64)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

# try using different optimizers and different optimizer configs
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

cat('Train...\n')
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = 4,
  validation_data = list(x_test, y_test)
)

  
