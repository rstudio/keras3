# Introduction to Keras for engineers

## Introduction

Keras 3 is a deep learning framework works with TensorFlow, JAX, and
PyTorch interchangeably. This notebook will walk you through key Keras 3
workflows.

Let’s start by installing Keras 3:

``` r
install.packages("keras3")
keras3::install_keras()
```

## Setup

We’re going to be using the tensorflow backend here – but you can edit
the string below to `"jax"` or `"torch"` and hit “Restart runtime”, and
the whole notebook will run just the same! This entire guide is
backend-agnostic.

``` r
library(tensorflow, exclude = c("shape", "set_random_seed"))
library(keras3)

# Note that you must configure the backend
# before calling any other keras functions.
# The backend cannot be changed once the
# package is imported.
use_backend("tensorflow")
```

## A first example: A MNIST convnet

Let’s start with the Hello World of ML: training a convnet to classify
MNIST digits.

Here’s the data:

``` r
# Load the data and split it between train and test sets
c(c(x_train, y_train), c(x_test, y_test)) %<-% keras3::dataset_mnist()

# Scale images to the [0, 1] range
x_train <- x_train / 255
x_test <- x_test / 255

# Make sure images have shape (28, 28, 1)
x_train <- array_reshape(x_train, c(-1, 28, 28, 1))
x_test <- array_reshape(x_test, c(-1, 28, 28, 1))

dim(x_train)
```

    ## [1] 60000    28    28     1

``` r
dim(x_test)
```

    ## [1] 10000    28    28     1

Here’s our model.

Different model-building options that Keras offers include:

- [The Sequential
  API](https://keras3.posit.co/articles/sequential_model.md) (what we
  use below)
- [The Functional
  API](https://keras3.posit.co/articles/functional_api.md) (most
  typical)
- [Writing your own models yourself via
  subclassing](https://keras3.posit.co/articles/making_new_layers_and_models_via_subclassing.md)
  (for advanced use cases)

``` r
# Model parameters
num_classes <- 10
input_shape <- c(28, 28, 1)

model <- keras_model_sequential(input_shape = input_shape)
model |>
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") |>
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") |>
  layer_max_pooling_2d(pool_size = c(2, 2)) |>
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") |>
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") |>
  layer_global_average_pooling_2d() |>
  layer_dropout(rate = 0.5) |>
  layer_dense(units = num_classes, activation = "softmax")
```

Here’s our model summary:

``` r
summary(model)
```

    ## Model: "sequential"
    ## ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ## ┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
    ## ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    ## │ conv2d (Conv2D)                 │ (None, 26, 26, 64)     │           640 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ conv2d_1 (Conv2D)               │ (None, 24, 24, 64)     │        36,928 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ max_pooling2d (MaxPooling2D)    │ (None, 12, 12, 64)     │             0 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ conv2d_2 (Conv2D)               │ (None, 10, 10, 128)    │        73,856 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ conv2d_3 (Conv2D)               │ (None, 8, 8, 128)      │       147,584 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ global_average_pooling2d        │ (None, 128)            │             0 │
    ## │ (GlobalAveragePooling2D)        │                        │               │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ dropout (Dropout)               │ (None, 128)            │             0 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ dense (Dense)                   │ (None, 10)             │         1,290 │
    ## └─────────────────────────────────┴────────────────────────┴───────────────┘
    ##  Total params: 260,298 (1016.79 KB)
    ##  Trainable params: 260,298 (1016.79 KB)
    ##  Non-trainable params: 0 (0.00 B)

We use the
[`compile()`](https://generics.r-lib.org/reference/compile.html) method
to specify the optimizer, loss function, and the metrics to monitor.
Note that with the JAX and TensorFlow backends, XLA compilation is
turned on by default.

``` r
model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = list(
    metric_sparse_categorical_accuracy(name = "acc")
  )
)
```

Let’s train and evaluate the model. We’ll set aside a validation split
of 15% of the data during training to monitor generalization on unseen
data.

``` r
batch_size <- 128
epochs <- 10

callbacks <- list(
  callback_model_checkpoint(filepath="model_at_epoch_{epoch}.keras"),
  callback_early_stopping(monitor="val_loss", patience=2)
)

model |> fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.15,
  callbacks = callbacks
)
```

    ## Epoch 1/10
    ## 399/399 - 7s - 17ms/step - acc: 0.7472 - loss: 0.7457 - val_acc: 0.9647 - val_loss: 0.1205
    ## Epoch 2/10
    ## 399/399 - 2s - 6ms/step - acc: 0.9377 - loss: 0.2080 - val_acc: 0.9770 - val_loss: 0.0784
    ## Epoch 3/10
    ## 399/399 - 2s - 6ms/step - acc: 0.9563 - loss: 0.1489 - val_acc: 0.9820 - val_loss: 0.0623
    ## Epoch 4/10
    ## 399/399 - 2s - 6ms/step - acc: 0.9645 - loss: 0.1190 - val_acc: 0.9858 - val_loss: 0.0492
    ## Epoch 5/10
    ## 399/399 - 2s - 6ms/step - acc: 0.9710 - loss: 0.1007 - val_acc: 0.9871 - val_loss: 0.0449
    ## Epoch 6/10
    ## 399/399 - 2s - 6ms/step - acc: 0.9750 - loss: 0.0873 - val_acc: 0.9881 - val_loss: 0.0395
    ## Epoch 7/10
    ## 399/399 - 2s - 6ms/step - acc: 0.9765 - loss: 0.0791 - val_acc: 0.9893 - val_loss: 0.0388
    ## Epoch 8/10
    ## 399/399 - 2s - 6ms/step - acc: 0.9789 - loss: 0.0693 - val_acc: 0.9877 - val_loss: 0.0420
    ## Epoch 9/10
    ## 399/399 - 2s - 6ms/step - acc: 0.9802 - loss: 0.0666 - val_acc: 0.9901 - val_loss: 0.0369
    ## Epoch 10/10
    ## 399/399 - 2s - 6ms/step - acc: 0.9823 - loss: 0.0589 - val_acc: 0.9911 - val_loss: 0.0339

``` r
score <- model |> evaluate(x_test, y_test, verbose = 0)
```

During training, we were saving a model at the end of each epoch. You
can also save the model in its latest state like this:

``` r
save_model(model, "final_model.keras", overwrite=TRUE)
```

And reload it like this:

``` r
model <- load_model("final_model.keras")
```

Next, you can query predictions of class probabilities with
[`predict()`](https://rdrr.io/r/stats/predict.html):

``` r
predictions <- model |> predict(x_test)
```

    ## 313/313 - 1s - 2ms/step

``` r
dim(predictions)
```

    ## [1] 10000    10

That’s it for the basics!

## Writing cross-framework custom components

Keras enables you to write custom Layers, Models, Metrics, Losses, and
Optimizers that work across TensorFlow, JAX, and PyTorch with the same
codebase. Let’s take a look at custom layers first.

The `op_` namespace contains:

- An implementation of the NumPy API, e.g. `op_stack` or `op_matmul`.
- A set of neural network specific ops that are absent from NumPy, such
  as `op_conv` or `op_binary_crossentropy`.

Let’s make a custom `Dense` layer that works with all backends:

``` r
layer_my_dense <- Layer(
  classname = "MyDense",
  initialize = function(units, activation = NULL, name = NULL, ...) {
    super$initialize(name = name, ...)
    self$units <- units
    self$activation <- activation
  },
  build = function(input_shape) {
    input_dim <- tail(input_shape, 1)
    self$w <- self$add_weight(
      shape = shape(input_dim, self$units),
      initializer = initializer_glorot_normal(),
      name = "kernel",
      trainable = TRUE
    )
    self$b <- self$add_weight(
      shape = shape(self$units),
      initializer = initializer_zeros(),
      name = "bias",
      trainable = TRUE
    )
  },
  call = function(inputs) {
    # Use Keras ops to create backend-agnostic layers/metrics/etc.
    x <- op_matmul(inputs, self$w) + self$b
    if (!is.null(self$activation))
      x <- self$activation(x)
    x
  }
)
```

Next, let’s make a custom `Dropout` layer that relies on the `random_*`
namespace:

``` r
layer_my_dropout <- Layer(
  "MyDropout",
  initialize = function(rate, name = NULL, seed = NULL, ...) {
    super$initialize(name = name)
    self$rate <- rate
    # Use seed_generator for managing RNG state.
    # It is a state element and its seed variable is
    # tracked as part of `layer$variables`.
    self$seed_generator <- random_seed_generator(seed)
  },
  call = function(inputs) {
    # Use `keras3::random_*` for random ops.
    random_dropout(inputs, self$rate, seed = self$seed_generator)
  }
)
```

Next, let’s write a custom subclassed model that uses our two custom
layers:

``` r
MyModel <- Model(
  "MyModel",
  initialize = function(num_classes, ...) {
    super$initialize(...)
    self$conv_base <-
      keras_model_sequential() |>
      layer_conv_2d(64, kernel_size = c(3, 3), activation = "relu") |>
      layer_conv_2d(64, kernel_size = c(3, 3), activation = "relu") |>
      layer_max_pooling_2d(pool_size = c(2, 2)) |>
      layer_conv_2d(128, kernel_size = c(3, 3), activation = "relu") |>
      layer_conv_2d(128, kernel_size = c(3, 3), activation = "relu") |>
      layer_global_average_pooling_2d()

    self$dp <- layer_my_dropout(rate = 0.5)
    self$dense <- layer_my_dense(units = num_classes,
                                 activation = activation_softmax)
  },
  call = function(inputs) {
    inputs |>
      self$conv_base() |>
      self$dp() |>
      self$dense()
  }
)
```

Let’s compile it and fit it:

``` r
model <- MyModel(num_classes = 10)
model |> compile(
  loss = loss_sparse_categorical_crossentropy(),
  optimizer = optimizer_adam(learning_rate = 1e-3),
  metrics = list(
    metric_sparse_categorical_accuracy(name = "acc")
  )
)

model |> fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = 1, # For speed
  validation_split = 0.15
)
```

    ## 399/399 - 6s - 16ms/step - acc: 0.7349 - loss: 0.7742 - val_acc: 0.9261 - val_loss: 0.2408

## Training models on arbitrary data sources

All Keras models can be trained and evaluated on a wide variety of data
sources, independently of the backend you’re using. This includes:

- Arrays
- Dataframes
- TensorFlow `tf_dataset` objects
- PyTorch `DataLoader` objects
- Keras `PyDataset` objects

They all work whether you’re using TensorFlow, JAX, or PyTorch as your
Keras backend.

Let’s try this out with `tf_dataset`:

``` r
library(tfdatasets, exclude = "shape")

train_dataset <- list(x_train, y_train) |>
  tensor_slices_dataset() |>
  dataset_batch(batch_size) |>
  dataset_prefetch(buffer_size = tf$data$AUTOTUNE)

test_dataset <- list(x_test, y_test) |>
  tensor_slices_dataset() |>
  dataset_batch(batch_size) |>
  dataset_prefetch(buffer_size = tf$data$AUTOTUNE)

model <- MyModel(num_classes = 10)
model |> compile(
  loss = loss_sparse_categorical_crossentropy(),
  optimizer = optimizer_adam(learning_rate = 1e-3),
  metrics = list(
    metric_sparse_categorical_accuracy(name = "acc")
  )
)

model |> fit(train_dataset, epochs = 1, validation_data = test_dataset)
```

    ## 469/469 - 7s - 16ms/step - acc: 0.7505 - loss: 0.7455 - val_acc: 0.9088 - val_loss: 0.3040

## Further reading

This concludes our short overview of the new multi-backend capabilities
of Keras 3. Next, you can learn about:

### How to customize what happens in `fit()`

Want to implement a non-standard training algorithm yourself but still
want to benefit from the power and usability of
[`fit()`](https://generics.r-lib.org/reference/fit.html)? It’s easy to
customize [`fit()`](https://generics.r-lib.org/reference/fit.html) to
support arbitrary use cases:

- [Customizing what happens in `fit()` with
  TensorFlow](https://keras3.posit.co/articles/custom_train_step_in_tensorflow.md)

## How to write custom training loops

- [Writing a training loop from scratch in
  TensorFlow](https://keras3.posit.co/articles/writing_a_custom_training_loop_in_tensorflow.md)

## How to distribute training

- [Guide to distributed training with
  TensorFlow](https://keras3.posit.co/articles/distributed_training_with_tensorflow.md)

Enjoy the library! 🚀
