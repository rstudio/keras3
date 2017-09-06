library(keras)

# create model
model <- keras_model_sequential()

# add layers and compile the model
model %>% 
  layer_dense(units = 32, activation = 'relu', input_shape = c(100)) %>% 
  layer_dense(units = 1, activation = 'sigmoid') %>% 
  compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )

# Generate dummy data
data <- matrix(runif(1000*100), nrow = 1000, ncol = 100)
labels <- matrix(round(runif(1000, min = 0, max = 1)), nrow = 1000, ncol = 1)

# create callbacks
callbacks <- list(
  callback_model_checkpoint("cbk_checkpoint.h5"),
  callback_csv_logger("cbk_history.csv")
)

if (is_backend("tensorflow"))
  callbacks <- append(callbacks, callback_tensorboard(log_dir = "tflogs"))

# Train the model, iterating on the data in batches of 32 samples
model %>% fit(
  data, 
  labels, 
  epochs=10, 
  batch_size=32, 
  validation_split = 0.2,
  callbacks = callbacks,
  view_metrics = FALSE
)

# Save model and weights
save_model_hdf5(model, "model.h5")
if (!utils::file_test("-d", "weights"))
  dir.create("weights")
save_model_weights_hdf5(model, file.path("weights", "weights.h5"))
