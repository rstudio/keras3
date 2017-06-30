#' Train and evaluate a simple MLP on the Reuters newswire topic classification task.

library(keras)

max_words <- 1000
batch_size <- 32
epochs <- 5

cat('Loading data...\n')
reuters <- dataset_reuters(num_words = max_words, test_split = 0.2)
x_train <- reuters$train$x
y_train <- reuters$train$y
x_test <- reuters$test$x
y_test <- reuters$test$y

cat(length(x_train), 'train sequences\n')
cat(length(x_test), 'test sequences\n')

num_classes <- max(y_train) + 1
cat(num_classes, '\n')

cat('Vectorizing sequence data...\n')

tokenizer <- text_tokenizer(num_words = max_words)
x_train <- sequences_to_matrix(tokenizer, x_train, mode = 'binary')
x_test <- sequences_to_matrix(tokenizer, x_test, mode = 'binary')

cat('x_train shape:', dim(x_train), '\n')
cat('x_test shape:', dim(x_test), '\n')

cat('Convert class vector to binary class matrix',
    '(for use with categorical_crossentropy)\n')
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)
cat('y_train shape:', dim(y_train), '\n')
cat('y_test shape:', dim(y_test), '\n')

cat('Building model...\n')
model <- keras_model_sequential()
model %>%
  layer_dense(units = 512, input_shape = c(max_words)) %>% 
  layer_activation(activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes) %>% 
  layer_activation(activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_split = 0.1
)

score <- model %>% evaluate(
  x_test, y_test,
  batch_size = batch_size,
  verbose = 1
)

cat('Test score:', score[[1]], '\n')
cat('Test accuracy', score[[2]], '\n')

