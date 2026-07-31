# Timeseries classification from scratch

## Introduction

This example shows how to do timeseries classification from scratch,
starting from raw CSV timeseries files on disk. We demonstrate the
workflow on the FordA dataset from the [UCR/UEA
archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).

## Setup

``` r
library(keras3)
use_backend("jax")
```

## Load the data: the FordA dataset

### Dataset description

The dataset we are using here is called FordA. The data comes from the
UCR archive. The dataset contains 3601 training instances and another
1320 testing instances. Each timeseries corresponds to a measurement of
engine noise captured by a motor sensor. For this task, the goal is to
automatically detect the presence of a specific issue with the engine.
The problem is a balanced binary classification task. The full
description of this dataset can be found
[here](http://www.j-wichard.de/publications/FordPaper.pdf).

### Read the TSV data

We will use the `FordA_TRAIN` file for training and the `FordA_TEST`
file for testing. The simplicity of this dataset allows us to
demonstrate effectively how to use ConvNets for timeseries
classification. In this file, the first column corresponds to the label.

``` r
get_data <- function(path) {
  if(path |> startsWith("https://"))
    path <- get_file(origin = path)  # cache file locally

  data <- readr::read_tsv(
    path, col_names = FALSE,
    # Each row is: one integer (the label),
    # followed by 500 doubles (the timeseries)
    col_types = paste0("i", strrep("d", 500))
  )

  y <- as.matrix(data[[1]])
  x <- as.matrix(data[,-1])
  dimnames(x) <- dimnames(y) <- NULL

  list(x, y)
}

root_url <- "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
c(x_train, y_train) %<-% get_data(paste0(root_url, "FordA_TRAIN.tsv"))
c(x_test, y_test) %<-% get_data(paste0(root_url, "FordA_TEST.tsv"))

str(keras3:::named_list(
  x_train, y_train,
  x_test, y_test
))
```

    ## List of 4
    ##  $ x_train: num [1:3601, 1:500] -0.797 0.805 0.728 -0.234 -0.171 ...
    ##  $ y_train: int [1:3601, 1] -1 1 -1 -1 -1 1 1 1 1 1 ...
    ##  $ x_test : num [1:1320, 1:500] -0.14 0.334 0.717 1.24 -1.159 ...
    ##  $ y_test : int [1:1320, 1] -1 -1 -1 1 -1 1 -1 -1 1 1 ...

## Visualize the data

Here we visualize one timeseries example for each class in the dataset.

``` r
plot(NULL, main = "Timeseries Data",
     xlab = "Timepoints",  ylab = "Values",
     xlim = c(1, ncol(x_test)),
     ylim = range(x_test))
grid()
lines(x_test[match(-1, y_test), ], col = "blue")
lines(x_test[match( 1, y_test), ], col = "red")
legend("topright", legend=c("label -1", "label 1"), col=c("blue", "red"), lty=1)
```

![Plot of Example Timeseries
Data](timeseries_classification_from_scratch/unnamed-chunk-3-1.png)

Plot of Example Timeseries Data

## Standardize the data

Our timeseries are already in a single length (500). However, their
values are usually in various ranges. This is not ideal for a neural
network; in general we should seek to make the input values normalized.
For this specific dataset, the data is already z-normalized: each
timeseries sample has a mean equal to zero and a standard deviation
equal to one. This type of normalization is very common for timeseries
classification problems, see [Bagnall et
al. (2016)](https://link.springer.com/article/10.1007/s10618-016-0483-9).

Note that the timeseries data used here are univariate, meaning we only
have one channel per timeseries example. We will therefore transform the
timeseries into a multivariate one with one channel using a simple
reshaping via numpy. This will allow us to construct a model that is
easily applicable to multivariate time series.

``` r
dim(x_train) <- c(dim(x_train), 1)
dim(x_test) <- c(dim(x_test), 1)
```

Finally, in order to use `sparse_categorical_crossentropy`, we will have
to count the number of classes beforehand.

``` r
num_classes <- length(unique(y_train))
```

Now we shuffle the training set because we will be using the
`validation_split` option later when training.

``` r
c(x_train, y_train) %<-% listarrays::shuffle_rows(x_train, y_train)
# idx <- sample.int(nrow(x_train))
# x_train %<>% .[idx,, ,drop = FALSE]
# y_train %<>% .[idx,  ,drop = FALSE]
```

Standardize the labels to positive integers. The expected labels will
then be 0 and 1.

``` r
y_train[y_train == -1L] <- 0L
y_test[y_test == -1L] <- 0L
```

## Build a model

We build a Fully Convolutional Neural Network originally proposed in
[this paper](https://arxiv.org/abs/1611.06455). The implementation is
based on the TF 2 version provided
[here](https://github.com/hfawaz/dl-4-tsc/). The following
hyperparameters (kernel_size, filters, the usage of BatchNorm) were
found via random search using
[KerasTuner](https://github.com/keras-team/keras-tuner).

``` r
make_model <- function(input_shape) {
  inputs <- keras_input(input_shape)

  outputs <- inputs |>
    # conv1
    layer_conv_1d(filters = 64, kernel_size = 3, padding = "same") |>
    layer_batch_normalization() |>
    layer_activation_relu() |>
    # conv2
    layer_conv_1d(filters = 64, kernel_size = 3, padding = "same") |>
    layer_batch_normalization() |>
    layer_activation_relu() |>
    # conv3
    layer_conv_1d(filters = 64, kernel_size = 3, padding = "same") |>
    layer_batch_normalization() |>
    layer_activation_relu() |>
    # pooling
    layer_global_average_pooling_1d() |>
    # final output
    layer_dense(num_classes, activation = "softmax")

  keras_model(inputs, outputs)
}

model <- make_model(input_shape = dim(x_train)[-1])
```

``` r
model
```

    ## Model: "functional"
    ## ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓
    ## ┃ Layer (type)                ┃ Output Shape          ┃    Param # ┃ Trai… ┃
    ## ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩
    ## │ input_layer (InputLayer)    │ (None, 500, 1)        │          0 │   -   │
    ## ├─────────────────────────────┼───────────────────────┼────────────┼───────┤
    ## │ conv1d (Conv1D)             │ (None, 500, 64)       │        256 │   Y   │
    ## ├─────────────────────────────┼───────────────────────┼────────────┼───────┤
    ## │ batch_normalization         │ (None, 500, 64)       │        256 │   Y   │
    ## │ (BatchNormalization)        │                       │            │       │
    ## ├─────────────────────────────┼───────────────────────┼────────────┼───────┤
    ## │ re_lu (ReLU)                │ (None, 500, 64)       │          0 │   -   │
    ## ├─────────────────────────────┼───────────────────────┼────────────┼───────┤
    ## │ conv1d_1 (Conv1D)           │ (None, 500, 64)       │     12,352 │   Y   │
    ## ├─────────────────────────────┼───────────────────────┼────────────┼───────┤
    ## │ batch_normalization_1       │ (None, 500, 64)       │        256 │   Y   │
    ## │ (BatchNormalization)        │                       │            │       │
    ## ├─────────────────────────────┼───────────────────────┼────────────┼───────┤
    ## │ re_lu_1 (ReLU)              │ (None, 500, 64)       │          0 │   -   │
    ## ├─────────────────────────────┼───────────────────────┼────────────┼───────┤
    ## │ conv1d_2 (Conv1D)           │ (None, 500, 64)       │     12,352 │   Y   │
    ## ├─────────────────────────────┼───────────────────────┼────────────┼───────┤
    ## │ batch_normalization_2       │ (None, 500, 64)       │        256 │   Y   │
    ## │ (BatchNormalization)        │                       │            │       │
    ## ├─────────────────────────────┼───────────────────────┼────────────┼───────┤
    ## │ re_lu_2 (ReLU)              │ (None, 500, 64)       │          0 │   -   │
    ## ├─────────────────────────────┼───────────────────────┼────────────┼───────┤
    ## │ global_average_pooling1d    │ (None, 64)            │          0 │   -   │
    ## │ (GlobalAveragePooling1D)    │                       │            │       │
    ## ├─────────────────────────────┼───────────────────────┼────────────┼───────┤
    ## │ dense (Dense)               │ (None, 2)             │        130 │   Y   │
    ## └─────────────────────────────┴───────────────────────┴────────────┴───────┘
    ##  Total params: 25,858 (101.01 KB)
    ##  Trainable params: 25,474 (99.51 KB)
    ##  Non-trainable params: 384 (1.50 KB)

``` r
plot(model, show_shapes = TRUE)
```

![plot of chunk
unnamed-chunk-9](timeseries_classification_from_scratch/unnamed-chunk-9-1.png)

plot of chunk unnamed-chunk-9

## Train the model

``` r
epochs <- 500
batch_size <- 32

callbacks <- c(
  callback_model_checkpoint(
    "best_model.keras", save_best_only = TRUE,
    monitor = "val_loss"
  ),
  callback_reduce_lr_on_plateau(
    monitor = "val_loss", factor = 0.5,
    patience = 20, min_lr = 0.0001
  ),
  callback_early_stopping(
    monitor = "val_loss", patience = 50,
    verbose = 1
  )
)


model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "sparse_categorical_accuracy"
)

history <- model |> fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  callbacks = callbacks,
  validation_split = 0.2
)
```

    ## Epoch 1/500
    ## 90/90 - 2s - 19ms/step - loss: 0.5311 - sparse_categorical_accuracy: 0.7205 - val_loss: 0.7772 - val_sparse_categorical_accuracy: 0.4896 - learning_rate: 0.0010
    ## Epoch 2/500
    ## 90/90 - 0s - 3ms/step - loss: 0.4819 - sparse_categorical_accuracy: 0.7653 - val_loss: 0.9960 - val_sparse_categorical_accuracy: 0.4896 - learning_rate: 0.0010
    ## Epoch 3/500
    ## 90/90 - 0s - 2ms/step - loss: 0.4509 - sparse_categorical_accuracy: 0.7660 - val_loss: 0.9115 - val_sparse_categorical_accuracy: 0.4896 - learning_rate: 0.0010
    ## Epoch 4/500
    ## 90/90 - 0s - 2ms/step - loss: 0.4086 - sparse_categorical_accuracy: 0.7986 - val_loss: 0.8081 - val_sparse_categorical_accuracy: 0.4951 - learning_rate: 0.0010
    ## Epoch 5/500
    ## 90/90 - 0s - 2ms/step - loss: 0.4159 - sparse_categorical_accuracy: 0.7927 - val_loss: 0.5280 - val_sparse_categorical_accuracy: 0.6824 - learning_rate: 0.0010
    ## Epoch 6/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3919 - sparse_categorical_accuracy: 0.8160 - val_loss: 0.4388 - val_sparse_categorical_accuracy: 0.7462 - learning_rate: 0.0010
    ## Epoch 7/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3834 - sparse_categorical_accuracy: 0.8156 - val_loss: 0.5788 - val_sparse_categorical_accuracy: 0.7129 - learning_rate: 0.0010
    ## Epoch 8/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3705 - sparse_categorical_accuracy: 0.8198 - val_loss: 0.3998 - val_sparse_categorical_accuracy: 0.7920 - learning_rate: 0.0010
    ## Epoch 9/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3656 - sparse_categorical_accuracy: 0.8271 - val_loss: 0.5670 - val_sparse_categorical_accuracy: 0.6810 - learning_rate: 0.0010
    ## Epoch 10/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3599 - sparse_categorical_accuracy: 0.8288 - val_loss: 0.4043 - val_sparse_categorical_accuracy: 0.8252 - learning_rate: 0.0010
    ## Epoch 11/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3498 - sparse_categorical_accuracy: 0.8368 - val_loss: 0.4675 - val_sparse_categorical_accuracy: 0.7601 - learning_rate: 0.0010
    ## Epoch 12/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3401 - sparse_categorical_accuracy: 0.8434 - val_loss: 0.8601 - val_sparse_categorical_accuracy: 0.5784 - learning_rate: 0.0010
    ## Epoch 13/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3264 - sparse_categorical_accuracy: 0.8615 - val_loss: 0.5353 - val_sparse_categorical_accuracy: 0.7143 - learning_rate: 0.0010
    ## Epoch 14/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3290 - sparse_categorical_accuracy: 0.8542 - val_loss: 0.4472 - val_sparse_categorical_accuracy: 0.7739 - learning_rate: 0.0010
    ## Epoch 15/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3229 - sparse_categorical_accuracy: 0.8535 - val_loss: 0.6410 - val_sparse_categorical_accuracy: 0.6560 - learning_rate: 0.0010
    ## Epoch 16/500
    ## 90/90 - 0s - 3ms/step - loss: 0.3178 - sparse_categorical_accuracy: 0.8639 - val_loss: 0.3649 - val_sparse_categorical_accuracy: 0.8391 - learning_rate: 0.0010
    ## Epoch 17/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3051 - sparse_categorical_accuracy: 0.8701 - val_loss: 0.5393 - val_sparse_categorical_accuracy: 0.7184 - learning_rate: 0.0010
    ## Epoch 18/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2980 - sparse_categorical_accuracy: 0.8736 - val_loss: 0.4979 - val_sparse_categorical_accuracy: 0.7309 - learning_rate: 0.0010
    ## Epoch 19/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3029 - sparse_categorical_accuracy: 0.8701 - val_loss: 0.3814 - val_sparse_categorical_accuracy: 0.8086 - learning_rate: 0.0010
    ## Epoch 20/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2953 - sparse_categorical_accuracy: 0.8774 - val_loss: 0.6389 - val_sparse_categorical_accuracy: 0.6727 - learning_rate: 0.0010
    ## Epoch 21/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2789 - sparse_categorical_accuracy: 0.8813 - val_loss: 0.3071 - val_sparse_categorical_accuracy: 0.8724 - learning_rate: 0.0010
    ## Epoch 22/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2921 - sparse_categorical_accuracy: 0.8781 - val_loss: 0.4132 - val_sparse_categorical_accuracy: 0.7933 - learning_rate: 0.0010
    ## Epoch 23/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2748 - sparse_categorical_accuracy: 0.8913 - val_loss: 0.3424 - val_sparse_categorical_accuracy: 0.8502 - learning_rate: 0.0010
    ## Epoch 24/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2717 - sparse_categorical_accuracy: 0.8875 - val_loss: 0.3340 - val_sparse_categorical_accuracy: 0.8488 - learning_rate: 0.0010
    ## Epoch 25/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2644 - sparse_categorical_accuracy: 0.8934 - val_loss: 0.2854 - val_sparse_categorical_accuracy: 0.8696 - learning_rate: 0.0010
    ## Epoch 26/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3095 - sparse_categorical_accuracy: 0.8653 - val_loss: 0.4604 - val_sparse_categorical_accuracy: 0.7725 - learning_rate: 0.0010
    ## Epoch 27/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2619 - sparse_categorical_accuracy: 0.9007 - val_loss: 0.2826 - val_sparse_categorical_accuracy: 0.8849 - learning_rate: 0.0010
    ## Epoch 28/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2768 - sparse_categorical_accuracy: 0.8813 - val_loss: 0.5080 - val_sparse_categorical_accuracy: 0.7712 - learning_rate: 0.0010
    ## Epoch 29/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2590 - sparse_categorical_accuracy: 0.8847 - val_loss: 0.2897 - val_sparse_categorical_accuracy: 0.8863 - learning_rate: 0.0010
    ## Epoch 30/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2558 - sparse_categorical_accuracy: 0.8948 - val_loss: 0.2923 - val_sparse_categorical_accuracy: 0.8641 - learning_rate: 0.0010
    ## Epoch 31/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2635 - sparse_categorical_accuracy: 0.8899 - val_loss: 0.2595 - val_sparse_categorical_accuracy: 0.8988 - learning_rate: 0.0010
    ## Epoch 32/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2535 - sparse_categorical_accuracy: 0.8924 - val_loss: 0.2624 - val_sparse_categorical_accuracy: 0.8974 - learning_rate: 0.0010
    ## Epoch 33/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2419 - sparse_categorical_accuracy: 0.9031 - val_loss: 0.2494 - val_sparse_categorical_accuracy: 0.8904 - learning_rate: 0.0010
    ## Epoch 34/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2365 - sparse_categorical_accuracy: 0.9073 - val_loss: 0.3469 - val_sparse_categorical_accuracy: 0.8252 - learning_rate: 0.0010
    ## Epoch 35/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2494 - sparse_categorical_accuracy: 0.8958 - val_loss: 0.3099 - val_sparse_categorical_accuracy: 0.8558 - learning_rate: 0.0010
    ## Epoch 36/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2468 - sparse_categorical_accuracy: 0.8990 - val_loss: 0.2652 - val_sparse_categorical_accuracy: 0.9015 - learning_rate: 0.0010
    ## Epoch 37/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2401 - sparse_categorical_accuracy: 0.9031 - val_loss: 0.2704 - val_sparse_categorical_accuracy: 0.8724 - learning_rate: 0.0010
    ## Epoch 38/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2416 - sparse_categorical_accuracy: 0.9021 - val_loss: 0.2682 - val_sparse_categorical_accuracy: 0.8835 - learning_rate: 0.0010
    ## Epoch 39/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2391 - sparse_categorical_accuracy: 0.9038 - val_loss: 0.4284 - val_sparse_categorical_accuracy: 0.8114 - learning_rate: 0.0010
    ## Epoch 40/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2311 - sparse_categorical_accuracy: 0.9024 - val_loss: 0.3154 - val_sparse_categorical_accuracy: 0.8627 - learning_rate: 0.0010
    ## Epoch 41/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2201 - sparse_categorical_accuracy: 0.9142 - val_loss: 0.2512 - val_sparse_categorical_accuracy: 0.8835 - learning_rate: 0.0010
    ## Epoch 42/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2165 - sparse_categorical_accuracy: 0.9198 - val_loss: 1.1002 - val_sparse_categorical_accuracy: 0.6200 - learning_rate: 0.0010
    ## Epoch 43/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2185 - sparse_categorical_accuracy: 0.9135 - val_loss: 0.3883 - val_sparse_categorical_accuracy: 0.8294 - learning_rate: 0.0010
    ## Epoch 44/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2146 - sparse_categorical_accuracy: 0.9153 - val_loss: 0.5537 - val_sparse_categorical_accuracy: 0.7476 - learning_rate: 0.0010
    ## Epoch 45/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2100 - sparse_categorical_accuracy: 0.9208 - val_loss: 0.3089 - val_sparse_categorical_accuracy: 0.8793 - learning_rate: 0.0010
    ## Epoch 46/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2159 - sparse_categorical_accuracy: 0.9115 - val_loss: 0.3610 - val_sparse_categorical_accuracy: 0.8363 - learning_rate: 0.0010
    ## Epoch 47/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1926 - sparse_categorical_accuracy: 0.9274 - val_loss: 0.6076 - val_sparse_categorical_accuracy: 0.7060 - learning_rate: 0.0010
    ## Epoch 48/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1938 - sparse_categorical_accuracy: 0.9240 - val_loss: 0.4014 - val_sparse_categorical_accuracy: 0.8141 - learning_rate: 0.0010
    ## Epoch 49/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1826 - sparse_categorical_accuracy: 0.9250 - val_loss: 0.2228 - val_sparse_categorical_accuracy: 0.9057 - learning_rate: 0.0010
    ## Epoch 50/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1795 - sparse_categorical_accuracy: 0.9337 - val_loss: 0.2706 - val_sparse_categorical_accuracy: 0.8766 - learning_rate: 0.0010
    ## Epoch 51/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1979 - sparse_categorical_accuracy: 0.9229 - val_loss: 0.2538 - val_sparse_categorical_accuracy: 0.8710 - learning_rate: 0.0010
    ## Epoch 52/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1623 - sparse_categorical_accuracy: 0.9434 - val_loss: 0.2913 - val_sparse_categorical_accuracy: 0.8655 - learning_rate: 0.0010
    ## Epoch 53/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1577 - sparse_categorical_accuracy: 0.9417 - val_loss: 0.2208 - val_sparse_categorical_accuracy: 0.9196 - learning_rate: 0.0010
    ## Epoch 54/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1462 - sparse_categorical_accuracy: 0.9559 - val_loss: 0.3119 - val_sparse_categorical_accuracy: 0.8405 - learning_rate: 0.0010
    ## Epoch 55/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1414 - sparse_categorical_accuracy: 0.9538 - val_loss: 0.7497 - val_sparse_categorical_accuracy: 0.7393 - learning_rate: 0.0010
    ## Epoch 56/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1408 - sparse_categorical_accuracy: 0.9535 - val_loss: 0.1804 - val_sparse_categorical_accuracy: 0.9307 - learning_rate: 0.0010
    ## Epoch 57/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1412 - sparse_categorical_accuracy: 0.9552 - val_loss: 0.3186 - val_sparse_categorical_accuracy: 0.8807 - learning_rate: 0.0010
    ## Epoch 58/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1236 - sparse_categorical_accuracy: 0.9590 - val_loss: 0.1839 - val_sparse_categorical_accuracy: 0.9293 - learning_rate: 0.0010
    ## Epoch 59/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1286 - sparse_categorical_accuracy: 0.9542 - val_loss: 0.3504 - val_sparse_categorical_accuracy: 0.8460 - learning_rate: 0.0010
    ## Epoch 60/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1395 - sparse_categorical_accuracy: 0.9528 - val_loss: 0.2896 - val_sparse_categorical_accuracy: 0.8932 - learning_rate: 0.0010
    ## Epoch 61/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1259 - sparse_categorical_accuracy: 0.9597 - val_loss: 0.4470 - val_sparse_categorical_accuracy: 0.8252 - learning_rate: 0.0010
    ## Epoch 62/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1262 - sparse_categorical_accuracy: 0.9601 - val_loss: 0.4769 - val_sparse_categorical_accuracy: 0.7684 - learning_rate: 0.0010
    ## Epoch 63/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1260 - sparse_categorical_accuracy: 0.9549 - val_loss: 0.5555 - val_sparse_categorical_accuracy: 0.7781 - learning_rate: 0.0010
    ## Epoch 64/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1255 - sparse_categorical_accuracy: 0.9615 - val_loss: 0.2608 - val_sparse_categorical_accuracy: 0.8974 - learning_rate: 0.0010
    ## Epoch 65/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1179 - sparse_categorical_accuracy: 0.9625 - val_loss: 0.3503 - val_sparse_categorical_accuracy: 0.8433 - learning_rate: 0.0010
    ## Epoch 66/500
    ## 90/90 - 0s - 3ms/step - loss: 0.1135 - sparse_categorical_accuracy: 0.9622 - val_loss: 0.1790 - val_sparse_categorical_accuracy: 0.9251 - learning_rate: 0.0010
    ## Epoch 67/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1088 - sparse_categorical_accuracy: 0.9653 - val_loss: 0.1605 - val_sparse_categorical_accuracy: 0.9307 - learning_rate: 0.0010
    ## Epoch 68/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1094 - sparse_categorical_accuracy: 0.9649 - val_loss: 0.1395 - val_sparse_categorical_accuracy: 0.9487 - learning_rate: 0.0010
    ## Epoch 69/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1201 - sparse_categorical_accuracy: 0.9615 - val_loss: 0.1484 - val_sparse_categorical_accuracy: 0.9334 - learning_rate: 0.0010
    ## Epoch 70/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1188 - sparse_categorical_accuracy: 0.9601 - val_loss: 0.2486 - val_sparse_categorical_accuracy: 0.8863 - learning_rate: 0.0010
    ## Epoch 71/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1124 - sparse_categorical_accuracy: 0.9632 - val_loss: 0.8323 - val_sparse_categorical_accuracy: 0.7282 - learning_rate: 0.0010
    ## Epoch 72/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1073 - sparse_categorical_accuracy: 0.9670 - val_loss: 0.1819 - val_sparse_categorical_accuracy: 0.9334 - learning_rate: 0.0010
    ## Epoch 73/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1027 - sparse_categorical_accuracy: 0.9674 - val_loss: 1.9502 - val_sparse_categorical_accuracy: 0.7018 - learning_rate: 0.0010
    ## Epoch 74/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1090 - sparse_categorical_accuracy: 0.9642 - val_loss: 0.5493 - val_sparse_categorical_accuracy: 0.8058 - learning_rate: 0.0010
    ## Epoch 75/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1167 - sparse_categorical_accuracy: 0.9618 - val_loss: 3.5464 - val_sparse_categorical_accuracy: 0.6893 - learning_rate: 0.0010
    ## Epoch 76/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1157 - sparse_categorical_accuracy: 0.9618 - val_loss: 2.6313 - val_sparse_categorical_accuracy: 0.6768 - learning_rate: 0.0010
    ## Epoch 77/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1043 - sparse_categorical_accuracy: 0.9660 - val_loss: 2.0556 - val_sparse_categorical_accuracy: 0.7295 - learning_rate: 0.0010
    ## Epoch 78/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1022 - sparse_categorical_accuracy: 0.9684 - val_loss: 0.7029 - val_sparse_categorical_accuracy: 0.7573 - learning_rate: 0.0010
    ## Epoch 79/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1063 - sparse_categorical_accuracy: 0.9684 - val_loss: 0.7888 - val_sparse_categorical_accuracy: 0.7406 - learning_rate: 0.0010
    ## Epoch 80/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1105 - sparse_categorical_accuracy: 0.9653 - val_loss: 0.5691 - val_sparse_categorical_accuracy: 0.7892 - learning_rate: 0.0010
    ## Epoch 81/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1049 - sparse_categorical_accuracy: 0.9649 - val_loss: 0.1545 - val_sparse_categorical_accuracy: 0.9473 - learning_rate: 0.0010
    ## Epoch 82/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0968 - sparse_categorical_accuracy: 0.9681 - val_loss: 0.1332 - val_sparse_categorical_accuracy: 0.9431 - learning_rate: 0.0010
    ## Epoch 83/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1064 - sparse_categorical_accuracy: 0.9632 - val_loss: 0.1697 - val_sparse_categorical_accuracy: 0.9307 - learning_rate: 0.0010
    ## Epoch 84/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1026 - sparse_categorical_accuracy: 0.9656 - val_loss: 0.1995 - val_sparse_categorical_accuracy: 0.9293 - learning_rate: 0.0010
    ## Epoch 85/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0974 - sparse_categorical_accuracy: 0.9691 - val_loss: 0.4980 - val_sparse_categorical_accuracy: 0.8322 - learning_rate: 0.0010
    ## Epoch 86/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0933 - sparse_categorical_accuracy: 0.9670 - val_loss: 0.2854 - val_sparse_categorical_accuracy: 0.8849 - learning_rate: 0.0010
    ## Epoch 87/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0999 - sparse_categorical_accuracy: 0.9639 - val_loss: 0.7666 - val_sparse_categorical_accuracy: 0.7212 - learning_rate: 0.0010
    ## Epoch 88/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1065 - sparse_categorical_accuracy: 0.9646 - val_loss: 0.4415 - val_sparse_categorical_accuracy: 0.8336 - learning_rate: 0.0010
    ## Epoch 89/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1026 - sparse_categorical_accuracy: 0.9622 - val_loss: 0.7326 - val_sparse_categorical_accuracy: 0.7351 - learning_rate: 0.0010
    ## Epoch 90/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1001 - sparse_categorical_accuracy: 0.9698 - val_loss: 0.1448 - val_sparse_categorical_accuracy: 0.9390 - learning_rate: 0.0010
    ## Epoch 91/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1087 - sparse_categorical_accuracy: 0.9670 - val_loss: 0.1277 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 0.0010
    ## Epoch 92/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1035 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.1785 - val_sparse_categorical_accuracy: 0.9279 - learning_rate: 0.0010
    ## Epoch 93/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0991 - sparse_categorical_accuracy: 0.9656 - val_loss: 0.5234 - val_sparse_categorical_accuracy: 0.7836 - learning_rate: 0.0010
    ## Epoch 94/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1044 - sparse_categorical_accuracy: 0.9649 - val_loss: 0.2801 - val_sparse_categorical_accuracy: 0.8821 - learning_rate: 0.0010
    ## Epoch 95/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0973 - sparse_categorical_accuracy: 0.9656 - val_loss: 0.6694 - val_sparse_categorical_accuracy: 0.7712 - learning_rate: 0.0010
    ## Epoch 96/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1099 - sparse_categorical_accuracy: 0.9604 - val_loss: 1.6304 - val_sparse_categorical_accuracy: 0.6047 - learning_rate: 0.0010
    ## Epoch 97/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1048 - sparse_categorical_accuracy: 0.9639 - val_loss: 0.1844 - val_sparse_categorical_accuracy: 0.9279 - learning_rate: 0.0010
    ## Epoch 98/500
    ## 90/90 - 0s - 3ms/step - loss: 0.0928 - sparse_categorical_accuracy: 0.9694 - val_loss: 0.1166 - val_sparse_categorical_accuracy: 0.9501 - learning_rate: 0.0010
    ## Epoch 99/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0942 - sparse_categorical_accuracy: 0.9660 - val_loss: 0.1493 - val_sparse_categorical_accuracy: 0.9459 - learning_rate: 0.0010
    ## Epoch 100/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0896 - sparse_categorical_accuracy: 0.9708 - val_loss: 1.8445 - val_sparse_categorical_accuracy: 0.6976 - learning_rate: 0.0010
    ## Epoch 101/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0847 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.4149 - val_sparse_categorical_accuracy: 0.8558 - learning_rate: 0.0010
    ## Epoch 102/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1024 - sparse_categorical_accuracy: 0.9670 - val_loss: 0.7227 - val_sparse_categorical_accuracy: 0.7420 - learning_rate: 0.0010
    ## Epoch 103/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0961 - sparse_categorical_accuracy: 0.9670 - val_loss: 1.1264 - val_sparse_categorical_accuracy: 0.6741 - learning_rate: 0.0010
    ## Epoch 104/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0986 - sparse_categorical_accuracy: 0.9625 - val_loss: 0.1876 - val_sparse_categorical_accuracy: 0.9251 - learning_rate: 0.0010
    ## Epoch 105/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0950 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.1346 - val_sparse_categorical_accuracy: 0.9473 - learning_rate: 0.0010
    ## Epoch 106/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0895 - sparse_categorical_accuracy: 0.9684 - val_loss: 1.2612 - val_sparse_categorical_accuracy: 0.7087 - learning_rate: 0.0010
    ## Epoch 107/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0930 - sparse_categorical_accuracy: 0.9656 - val_loss: 0.6277 - val_sparse_categorical_accuracy: 0.8072 - learning_rate: 0.0010
    ## Epoch 108/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0918 - sparse_categorical_accuracy: 0.9674 - val_loss: 0.5959 - val_sparse_categorical_accuracy: 0.8086 - learning_rate: 0.0010
    ## Epoch 109/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0940 - sparse_categorical_accuracy: 0.9708 - val_loss: 0.2216 - val_sparse_categorical_accuracy: 0.9043 - learning_rate: 0.0010
    ## Epoch 110/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1034 - sparse_categorical_accuracy: 0.9660 - val_loss: 0.2706 - val_sparse_categorical_accuracy: 0.8960 - learning_rate: 0.0010
    ## Epoch 111/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0867 - sparse_categorical_accuracy: 0.9705 - val_loss: 1.5376 - val_sparse_categorical_accuracy: 0.7087 - learning_rate: 0.0010
    ## Epoch 112/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0883 - sparse_categorical_accuracy: 0.9708 - val_loss: 0.4087 - val_sparse_categorical_accuracy: 0.8571 - learning_rate: 0.0010
    ## Epoch 113/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0829 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.1545 - val_sparse_categorical_accuracy: 0.9473 - learning_rate: 0.0010
    ## Epoch 114/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1058 - sparse_categorical_accuracy: 0.9660 - val_loss: 0.2307 - val_sparse_categorical_accuracy: 0.8974 - learning_rate: 0.0010
    ## Epoch 115/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0858 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.6216 - val_sparse_categorical_accuracy: 0.7282 - learning_rate: 0.0010
    ## Epoch 116/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0864 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.1994 - val_sparse_categorical_accuracy: 0.8960 - learning_rate: 0.0010
    ## Epoch 117/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0821 - sparse_categorical_accuracy: 0.9715 - val_loss: 0.2627 - val_sparse_categorical_accuracy: 0.9001 - learning_rate: 0.0010
    ## Epoch 118/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0946 - sparse_categorical_accuracy: 0.9653 - val_loss: 0.2622 - val_sparse_categorical_accuracy: 0.9015 - learning_rate: 0.0010
    ## Epoch 119/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0790 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.1385 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 5.0000e-04
    ## Epoch 120/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0805 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.1515 - val_sparse_categorical_accuracy: 0.9376 - learning_rate: 5.0000e-04
    ## Epoch 121/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0748 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.1198 - val_sparse_categorical_accuracy: 0.9501 - learning_rate: 5.0000e-04
    ## Epoch 122/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0767 - sparse_categorical_accuracy: 0.9729 - val_loss: 0.1050 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 5.0000e-04
    ## Epoch 123/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0743 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.3959 - val_sparse_categorical_accuracy: 0.8571 - learning_rate: 5.0000e-04
    ## Epoch 124/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0830 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.1226 - val_sparse_categorical_accuracy: 0.9473 - learning_rate: 5.0000e-04
    ## Epoch 125/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0788 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.1129 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 5.0000e-04
    ## Epoch 126/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0744 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.2572 - val_sparse_categorical_accuracy: 0.9001 - learning_rate: 5.0000e-04
    ## Epoch 127/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0746 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.2241 - val_sparse_categorical_accuracy: 0.9112 - learning_rate: 5.0000e-04
    ## Epoch 128/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0787 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.5435 - val_sparse_categorical_accuracy: 0.8336 - learning_rate: 5.0000e-04
    ## Epoch 129/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0739 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.4766 - val_sparse_categorical_accuracy: 0.8502 - learning_rate: 5.0000e-04
    ## Epoch 130/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0832 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.1707 - val_sparse_categorical_accuracy: 0.9307 - learning_rate: 5.0000e-04
    ## Epoch 131/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0690 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.1688 - val_sparse_categorical_accuracy: 0.9320 - learning_rate: 5.0000e-04
    ## Epoch 132/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0716 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.2072 - val_sparse_categorical_accuracy: 0.9182 - learning_rate: 5.0000e-04
    ## Epoch 133/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0669 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.3460 - val_sparse_categorical_accuracy: 0.8738 - learning_rate: 5.0000e-04
    ## Epoch 134/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0734 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.2999 - val_sparse_categorical_accuracy: 0.9001 - learning_rate: 5.0000e-04
    ## Epoch 135/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0722 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.1171 - val_sparse_categorical_accuracy: 0.9501 - learning_rate: 5.0000e-04
    ## Epoch 136/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0767 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.1302 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 5.0000e-04
    ## Epoch 137/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0797 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.1077 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 5.0000e-04
    ## Epoch 138/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0734 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.1014 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 5.0000e-04
    ## Epoch 139/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0808 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.1094 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 5.0000e-04
    ## Epoch 140/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0700 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.1236 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 5.0000e-04
    ## Epoch 141/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0710 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.2049 - val_sparse_categorical_accuracy: 0.9223 - learning_rate: 5.0000e-04
    ## Epoch 142/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0695 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.2553 - val_sparse_categorical_accuracy: 0.8988 - learning_rate: 5.0000e-04
    ## Epoch 143/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0754 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.4438 - val_sparse_categorical_accuracy: 0.8391 - learning_rate: 5.0000e-04
    ## Epoch 144/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0689 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.1173 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 5.0000e-04
    ## Epoch 145/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0683 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.1027 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 5.0000e-04
    ## Epoch 146/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0717 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.1868 - val_sparse_categorical_accuracy: 0.9237 - learning_rate: 5.0000e-04
    ## Epoch 147/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0759 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.1356 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 5.0000e-04
    ## Epoch 148/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0819 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.0974 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 5.0000e-04
    ## Epoch 149/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0691 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.1092 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 5.0000e-04
    ## Epoch 150/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0710 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.1969 - val_sparse_categorical_accuracy: 0.9293 - learning_rate: 5.0000e-04
    ## Epoch 151/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0694 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.1461 - val_sparse_categorical_accuracy: 0.9376 - learning_rate: 5.0000e-04
    ## Epoch 152/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0693 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.1916 - val_sparse_categorical_accuracy: 0.9168 - learning_rate: 5.0000e-04
    ## Epoch 153/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0697 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.1140 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 5.0000e-04
    ## Epoch 154/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0654 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.1158 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 5.0000e-04
    ## Epoch 155/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0695 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.1063 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 5.0000e-04
    ## Epoch 156/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0761 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.1101 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 5.0000e-04
    ## Epoch 157/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0701 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.1052 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 5.0000e-04
    ## Epoch 158/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0690 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.1441 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 5.0000e-04
    ## Epoch 159/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0725 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.1465 - val_sparse_categorical_accuracy: 0.9417 - learning_rate: 5.0000e-04
    ## Epoch 160/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0712 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.2777 - val_sparse_categorical_accuracy: 0.8877 - learning_rate: 5.0000e-04
    ## Epoch 161/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0690 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.1731 - val_sparse_categorical_accuracy: 0.9293 - learning_rate: 5.0000e-04
    ## Epoch 162/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0667 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.5835 - val_sparse_categorical_accuracy: 0.8169 - learning_rate: 5.0000e-04
    ## Epoch 163/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0608 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.1137 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 5.0000e-04
    ## Epoch 164/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0682 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.1408 - val_sparse_categorical_accuracy: 0.9487 - learning_rate: 5.0000e-04
    ## Epoch 165/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0652 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.7778 - val_sparse_categorical_accuracy: 0.7850 - learning_rate: 5.0000e-04
    ## Epoch 166/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0656 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.2635 - val_sparse_categorical_accuracy: 0.8974 - learning_rate: 5.0000e-04
    ## Epoch 167/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0665 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.2862 - val_sparse_categorical_accuracy: 0.9015 - learning_rate: 5.0000e-04
    ## Epoch 168/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0811 - sparse_categorical_accuracy: 0.9698 - val_loss: 0.3258 - val_sparse_categorical_accuracy: 0.8849 - learning_rate: 5.0000e-04
    ## Epoch 169/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0662 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.1006 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 2.5000e-04
    ## Epoch 170/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0621 - sparse_categorical_accuracy: 0.9813 - val_loss: 0.0987 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 2.5000e-04
    ## Epoch 171/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0581 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1245 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 2.5000e-04
    ## Epoch 172/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0613 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.2282 - val_sparse_categorical_accuracy: 0.9112 - learning_rate: 2.5000e-04
    ## Epoch 173/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0569 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.1198 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 2.5000e-04
    ## Epoch 174/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0609 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.1068 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 2.5000e-04
    ## Epoch 175/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0613 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.1311 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 2.5000e-04
    ## Epoch 176/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0578 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.0947 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 2.5000e-04
    ## Epoch 177/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0569 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.1496 - val_sparse_categorical_accuracy: 0.9473 - learning_rate: 2.5000e-04
    ## Epoch 178/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0572 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.1237 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 2.5000e-04
    ## Epoch 179/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0571 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.1104 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 2.5000e-04
    ## Epoch 180/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0576 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.1230 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 2.5000e-04
    ## Epoch 181/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0599 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0998 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
    ## Epoch 182/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0564 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0987 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 2.5000e-04
    ## Epoch 183/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0590 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.1230 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 2.5000e-04
    ## Epoch 184/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0559 - sparse_categorical_accuracy: 0.9813 - val_loss: 0.1231 - val_sparse_categorical_accuracy: 0.9473 - learning_rate: 2.5000e-04
    ## Epoch 185/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0581 - sparse_categorical_accuracy: 0.9813 - val_loss: 0.1365 - val_sparse_categorical_accuracy: 0.9501 - learning_rate: 2.5000e-04
    ## Epoch 186/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0634 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.1118 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 2.5000e-04
    ## Epoch 187/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0580 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.1043 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 2.5000e-04
    ## Epoch 188/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0566 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.1023 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
    ## Epoch 189/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0537 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.1730 - val_sparse_categorical_accuracy: 0.9417 - learning_rate: 2.5000e-04
    ## Epoch 190/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0630 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.1978 - val_sparse_categorical_accuracy: 0.9265 - learning_rate: 2.5000e-04
    ## Epoch 191/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0550 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.0979 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 2.5000e-04
    ## Epoch 192/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0605 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.0973 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 2.5000e-04
    ## Epoch 193/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0565 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.1702 - val_sparse_categorical_accuracy: 0.9445 - learning_rate: 2.5000e-04
    ## Epoch 194/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0552 - sparse_categorical_accuracy: 0.9813 - val_loss: 0.1879 - val_sparse_categorical_accuracy: 0.9307 - learning_rate: 2.5000e-04
    ## Epoch 195/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0562 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.1599 - val_sparse_categorical_accuracy: 0.9362 - learning_rate: 2.5000e-04
    ## Epoch 196/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0590 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.0983 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 2.5000e-04
    ## Epoch 197/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0525 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0942 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.2500e-04
    ## Epoch 198/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0511 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.1003 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
    ## Epoch 199/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0547 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.1013 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 1.2500e-04
    ## Epoch 200/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0539 - sparse_categorical_accuracy: 0.9813 - val_loss: 0.0978 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 1.2500e-04
    ## Epoch 201/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0514 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.1155 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
    ## Epoch 202/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0513 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0962 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
    ## Epoch 203/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0525 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0950 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
    ## Epoch 204/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0479 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.1140 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.2500e-04
    ## Epoch 205/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0525 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.1014 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 1.2500e-04
    ## Epoch 206/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0517 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.1005 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.2500e-04
    ## Epoch 207/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0558 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0964 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 1.2500e-04
    ## Epoch 208/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0517 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0997 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.2500e-04
    ## Epoch 209/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0503 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.1494 - val_sparse_categorical_accuracy: 0.9473 - learning_rate: 1.2500e-04
    ## Epoch 210/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0535 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0961 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.2500e-04
    ## Epoch 211/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0516 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0942 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 1.2500e-04
    ## Epoch 212/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0520 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0969 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.2500e-04
    ## Epoch 213/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0524 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0976 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.2500e-04
    ## Epoch 214/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0523 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0931 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.2500e-04
    ## Epoch 215/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0514 - sparse_categorical_accuracy: 0.9813 - val_loss: 0.0995 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.2500e-04
    ## Epoch 216/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0483 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.1015 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.2500e-04
    ## Epoch 217/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0509 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0988 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.2500e-04
    ## Epoch 218/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0549 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.1091 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 1.2500e-04
    ## Epoch 219/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0497 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.1090 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.2500e-04
    ## Epoch 220/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0499 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0943 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
    ## Epoch 221/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0512 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0953 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.2500e-04
    ## Epoch 222/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0517 - sparse_categorical_accuracy: 0.9813 - val_loss: 0.0939 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.2500e-04
    ## Epoch 223/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0521 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.1086 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
    ## Epoch 224/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0469 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0938 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.2500e-04
    ## Epoch 225/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0526 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.1065 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.2500e-04
    ## Epoch 226/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0484 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0991 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.2500e-04
    ## Epoch 227/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0525 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0932 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.2500e-04
    ## Epoch 228/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0473 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0959 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 1.2500e-04
    ## Epoch 229/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0494 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0929 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 1.2500e-04
    ## Epoch 230/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0481 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0992 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 1.2500e-04
    ## Epoch 231/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0529 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0960 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 1.2500e-04
    ## Epoch 232/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0495 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.1064 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
    ## Epoch 233/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0496 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0950 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 1.2500e-04
    ## Epoch 234/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0507 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.1121 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.2500e-04
    ## Epoch 235/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0515 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0962 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.2500e-04
    ## Epoch 236/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0500 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0929 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.2500e-04
    ## Epoch 237/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0513 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.1432 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 1.2500e-04
    ## Epoch 238/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0497 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.1256 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 1.2500e-04
    ## Epoch 239/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0485 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0959 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.2500e-04
    ## Epoch 240/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0484 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.1277 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 1.2500e-04
    ## Epoch 241/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0508 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.1071 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 1.2500e-04
    ## Epoch 242/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0508 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.1484 - val_sparse_categorical_accuracy: 0.9459 - learning_rate: 1.2500e-04
    ## Epoch 243/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0517 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0956 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 1.2500e-04
    ## Epoch 244/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0522 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.0975 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
    ## Epoch 245/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0498 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.1135 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.2500e-04
    ## Epoch 246/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0518 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0997 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.2500e-04
    ## Epoch 247/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0461 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0942 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.2500e-04
    ## Epoch 248/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0448 - sparse_categorical_accuracy: 0.9878 - val_loss: 0.0936 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
    ## Epoch 249/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0510 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.1023 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.2500e-04
    ## Epoch 250/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0498 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0971 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 1.0000e-04
    ## Epoch 251/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0492 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0971 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 1.0000e-04
    ## Epoch 252/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0511 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0987 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
    ## Epoch 253/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0503 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0922 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.0000e-04
    ## Epoch 254/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0500 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.1165 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 1.0000e-04
    ## Epoch 255/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0479 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0932 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
    ## Epoch 256/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0489 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0953 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.0000e-04
    ## Epoch 257/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0473 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0992 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
    ## Epoch 258/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0459 - sparse_categorical_accuracy: 0.9878 - val_loss: 0.0960 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.0000e-04
    ## Epoch 259/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0446 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0949 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 1.0000e-04
    ## Epoch 260/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0463 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0933 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.0000e-04
    ## Epoch 261/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0496 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.1472 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 1.0000e-04
    ## Epoch 262/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0466 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.1424 - val_sparse_categorical_accuracy: 0.9501 - learning_rate: 1.0000e-04
    ## Epoch 263/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0448 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0965 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 1.0000e-04
    ## Epoch 264/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0499 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.1111 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
    ## Epoch 265/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0530 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.0962 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
    ## Epoch 266/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0477 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0946 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.0000e-04
    ## Epoch 267/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0474 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0963 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 1.0000e-04
    ## Epoch 268/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0448 - sparse_categorical_accuracy: 0.9882 - val_loss: 0.0995 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
    ## Epoch 269/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0492 - sparse_categorical_accuracy: 0.9875 - val_loss: 0.0968 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.0000e-04
    ## Epoch 270/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0422 - sparse_categorical_accuracy: 0.9892 - val_loss: 0.0960 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.0000e-04
    ## Epoch 271/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0457 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0931 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.0000e-04
    ## Epoch 272/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0460 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0972 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.0000e-04
    ## Epoch 273/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0478 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.1007 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.0000e-04
    ## Epoch 274/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0446 - sparse_categorical_accuracy: 0.9878 - val_loss: 0.0957 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.0000e-04
    ## Epoch 275/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0427 - sparse_categorical_accuracy: 0.9882 - val_loss: 0.1211 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.0000e-04
    ## Epoch 276/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0474 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0974 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.0000e-04
    ## Epoch 277/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0462 - sparse_categorical_accuracy: 0.9882 - val_loss: 0.0948 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
    ## Epoch 278/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0465 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.1150 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.0000e-04
    ## Epoch 279/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0467 - sparse_categorical_accuracy: 0.9872 - val_loss: 0.0923 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
    ## Epoch 280/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0466 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0925 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
    ## Epoch 281/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0445 - sparse_categorical_accuracy: 0.9875 - val_loss: 0.1050 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
    ## Epoch 282/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0431 - sparse_categorical_accuracy: 0.9878 - val_loss: 0.0979 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
    ## Epoch 283/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0480 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0937 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.0000e-04
    ## Epoch 284/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0491 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0996 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.0000e-04
    ## Epoch 285/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0463 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.1123 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.0000e-04
    ## Epoch 286/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0454 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.1086 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.0000e-04
    ## Epoch 287/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0441 - sparse_categorical_accuracy: 0.9875 - val_loss: 0.0950 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.0000e-04
    ## Epoch 288/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0473 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.1221 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.0000e-04
    ## Epoch 289/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0461 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.1031 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
    ## Epoch 290/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0479 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0982 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
    ## Epoch 291/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0437 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0938 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.0000e-04
    ## Epoch 292/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0503 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0984 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 1.0000e-04
    ## Epoch 293/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0479 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0971 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.0000e-04
    ## Epoch 294/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0460 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0967 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.0000e-04
    ## Epoch 295/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0518 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0967 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.0000e-04
    ## Epoch 296/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0492 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0985 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.0000e-04
    ## Epoch 297/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0489 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0961 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
    ## Epoch 298/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0474 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0955 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 1.0000e-04
    ## Epoch 299/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0415 - sparse_categorical_accuracy: 0.9889 - val_loss: 0.1003 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
    ## Epoch 300/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0426 - sparse_categorical_accuracy: 0.9872 - val_loss: 0.0987 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 1.0000e-04
    ## Epoch 301/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0459 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0955 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
    ## Epoch 302/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0416 - sparse_categorical_accuracy: 0.9885 - val_loss: 0.1287 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 1.0000e-04
    ## Epoch 303/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0428 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0966 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.0000e-04
    ## Epoch 303: early stopping

## Evaluate model on test data

``` r
model <- load_model("best_model.keras")

results <- model |> evaluate(x_test, y_test)
```

    ## 42/42 - 0s - 9ms/step - loss: 0.0908 - sparse_categorical_accuracy: 0.9712

``` r
str(results)
```

    ## List of 2
    ##  $ loss                       : num 0.0908
    ##  $ sparse_categorical_accuracy: num 0.971

``` r
cat(
  "Test accuracy: ", results$sparse_categorical_accuracy, "\n",
  "Test loss: ", results$loss, "\n",
  sep = ""
)
```

    ## Test accuracy: 0.9712121
    ## Test loss: 0.09083591

## Plot the model’s training history

``` r
plot(history)
```

![Plot of Training History
Metrics](timeseries_classification_from_scratch/unnamed-chunk-12-1.png)

Plot of Training History Metrics

Plot just the training and validation accuracy:

``` r
plot(history, metric = "sparse_categorical_accuracy") +
  # scale x axis to actual number of epochs run before early stopping
  ggplot2::xlim(0, length(history$metrics$loss))
```

![Plot of Accuracy During
Training](timeseries_classification_from_scratch/unnamed-chunk-13-1.png)

Plot of Accuracy During Training

We can see how the training accuracy reaches almost 0.95 after 100
epochs. However, by observing the validation accuracy we can see how the
network still needs training until it reaches almost 0.97 for both the
validation and the training accuracy after 200 epochs. Beyond the 200th
epoch, if we continue on training, the validation accuracy will start
decreasing while the training accuracy will continue on increasing: the
model starts overfitting.
