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
    ## 90/90 - 3s - 33ms/step - loss: 0.5306 - sparse_categorical_accuracy: 0.7208 - val_loss: 0.7875 - val_sparse_categorical_accuracy: 0.4896 - learning_rate: 0.0010
    ## Epoch 2/500
    ## 90/90 - 0s - 2ms/step - loss: 0.4804 - sparse_categorical_accuracy: 0.7573 - val_loss: 0.9833 - val_sparse_categorical_accuracy: 0.4896 - learning_rate: 0.0010
    ## Epoch 3/500
    ## 90/90 - 0s - 2ms/step - loss: 0.4685 - sparse_categorical_accuracy: 0.7646 - val_loss: 0.9397 - val_sparse_categorical_accuracy: 0.4896 - learning_rate: 0.0010
    ## Epoch 4/500
    ## 90/90 - 0s - 2ms/step - loss: 0.4125 - sparse_categorical_accuracy: 0.7951 - val_loss: 0.7498 - val_sparse_categorical_accuracy: 0.4965 - learning_rate: 0.0010
    ## Epoch 5/500
    ## 90/90 - 0s - 2ms/step - loss: 0.4184 - sparse_categorical_accuracy: 0.7868 - val_loss: 0.4943 - val_sparse_categorical_accuracy: 0.8003 - learning_rate: 0.0010
    ## Epoch 6/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3982 - sparse_categorical_accuracy: 0.8080 - val_loss: 0.4219 - val_sparse_categorical_accuracy: 0.7822 - learning_rate: 0.0010
    ## Epoch 7/500
    ## 90/90 - 0s - 1ms/step - loss: 0.3889 - sparse_categorical_accuracy: 0.8125 - val_loss: 0.5938 - val_sparse_categorical_accuracy: 0.7046 - learning_rate: 0.0010
    ## Epoch 8/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3762 - sparse_categorical_accuracy: 0.8174 - val_loss: 0.3998 - val_sparse_categorical_accuracy: 0.7920 - learning_rate: 0.0010
    ## Epoch 9/500
    ## 90/90 - 0s - 1ms/step - loss: 0.3741 - sparse_categorical_accuracy: 0.8229 - val_loss: 0.6581 - val_sparse_categorical_accuracy: 0.6727 - learning_rate: 0.0010
    ## Epoch 10/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3659 - sparse_categorical_accuracy: 0.8257 - val_loss: 0.3532 - val_sparse_categorical_accuracy: 0.8377 - learning_rate: 0.0010
    ## Epoch 11/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3674 - sparse_categorical_accuracy: 0.8274 - val_loss: 0.6846 - val_sparse_categorical_accuracy: 0.7157 - learning_rate: 0.0010
    ## Epoch 12/500
    ## 90/90 - 0s - 1ms/step - loss: 0.3564 - sparse_categorical_accuracy: 0.8333 - val_loss: 0.5639 - val_sparse_categorical_accuracy: 0.7046 - learning_rate: 0.0010
    ## Epoch 13/500
    ## 90/90 - 0s - 1ms/step - loss: 0.3352 - sparse_categorical_accuracy: 0.8594 - val_loss: 0.4646 - val_sparse_categorical_accuracy: 0.7670 - learning_rate: 0.0010
    ## Epoch 14/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3351 - sparse_categorical_accuracy: 0.8479 - val_loss: 0.3448 - val_sparse_categorical_accuracy: 0.8502 - learning_rate: 0.0010
    ## Epoch 15/500
    ## 90/90 - 0s - 1ms/step - loss: 0.3273 - sparse_categorical_accuracy: 0.8524 - val_loss: 1.0034 - val_sparse_categorical_accuracy: 0.5465 - learning_rate: 0.0010
    ## Epoch 16/500
    ## 90/90 - 0s - 1ms/step - loss: 0.3293 - sparse_categorical_accuracy: 0.8590 - val_loss: 0.3876 - val_sparse_categorical_accuracy: 0.7712 - learning_rate: 0.0010
    ## Epoch 17/500
    ## 90/90 - 0s - 1ms/step - loss: 0.3128 - sparse_categorical_accuracy: 0.8628 - val_loss: 0.5153 - val_sparse_categorical_accuracy: 0.7184 - learning_rate: 0.0010
    ## Epoch 18/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3140 - sparse_categorical_accuracy: 0.8580 - val_loss: 0.3331 - val_sparse_categorical_accuracy: 0.8544 - learning_rate: 0.0010
    ## Epoch 19/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3067 - sparse_categorical_accuracy: 0.8694 - val_loss: 0.3217 - val_sparse_categorical_accuracy: 0.8433 - learning_rate: 0.0010
    ## Epoch 20/500
    ## 90/90 - 0s - 1ms/step - loss: 0.3004 - sparse_categorical_accuracy: 0.8747 - val_loss: 0.3337 - val_sparse_categorical_accuracy: 0.8641 - learning_rate: 0.0010
    ## Epoch 21/500
    ## 90/90 - 0s - 1ms/step - loss: 0.2862 - sparse_categorical_accuracy: 0.8788 - val_loss: 0.7405 - val_sparse_categorical_accuracy: 0.6394 - learning_rate: 0.0010
    ## Epoch 22/500
    ## 90/90 - 0s - 1ms/step - loss: 0.2957 - sparse_categorical_accuracy: 0.8795 - val_loss: 0.3662 - val_sparse_categorical_accuracy: 0.8322 - learning_rate: 0.0010
    ## Epoch 23/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2784 - sparse_categorical_accuracy: 0.8875 - val_loss: 0.3697 - val_sparse_categorical_accuracy: 0.8128 - learning_rate: 0.0010
    ## Epoch 24/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2776 - sparse_categorical_accuracy: 0.8847 - val_loss: 0.4496 - val_sparse_categorical_accuracy: 0.7628 - learning_rate: 0.0010
    ## Epoch 25/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2695 - sparse_categorical_accuracy: 0.8941 - val_loss: 0.3137 - val_sparse_categorical_accuracy: 0.8599 - learning_rate: 0.0010
    ## Epoch 26/500
    ## 90/90 - 0s - 2ms/step - loss: 0.3135 - sparse_categorical_accuracy: 0.8635 - val_loss: 0.9370 - val_sparse_categorical_accuracy: 0.6200 - learning_rate: 0.0010
    ## Epoch 27/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2724 - sparse_categorical_accuracy: 0.8882 - val_loss: 0.2878 - val_sparse_categorical_accuracy: 0.8863 - learning_rate: 0.0010
    ## Epoch 28/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2890 - sparse_categorical_accuracy: 0.8774 - val_loss: 0.8003 - val_sparse_categorical_accuracy: 0.6574 - learning_rate: 0.0010
    ## Epoch 29/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2679 - sparse_categorical_accuracy: 0.8837 - val_loss: 0.3961 - val_sparse_categorical_accuracy: 0.8211 - learning_rate: 0.0010
    ## Epoch 30/500
    ## 90/90 - 0s - 1ms/step - loss: 0.2643 - sparse_categorical_accuracy: 0.8917 - val_loss: 0.7993 - val_sparse_categorical_accuracy: 0.6602 - learning_rate: 0.0010
    ## Epoch 31/500
    ## 90/90 - 0s - 1ms/step - loss: 0.2653 - sparse_categorical_accuracy: 0.8934 - val_loss: 0.3430 - val_sparse_categorical_accuracy: 0.8322 - learning_rate: 0.0010
    ## Epoch 32/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2626 - sparse_categorical_accuracy: 0.8875 - val_loss: 0.3982 - val_sparse_categorical_accuracy: 0.7892 - learning_rate: 0.0010
    ## Epoch 33/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2459 - sparse_categorical_accuracy: 0.9038 - val_loss: 0.4210 - val_sparse_categorical_accuracy: 0.7698 - learning_rate: 0.0010
    ## Epoch 34/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2403 - sparse_categorical_accuracy: 0.9035 - val_loss: 0.2501 - val_sparse_categorical_accuracy: 0.8946 - learning_rate: 0.0010
    ## Epoch 35/500
    ## 90/90 - 0s - 1ms/step - loss: 0.2540 - sparse_categorical_accuracy: 0.8924 - val_loss: 0.4214 - val_sparse_categorical_accuracy: 0.7725 - learning_rate: 0.0010
    ## Epoch 36/500
    ## 90/90 - 0s - 1ms/step - loss: 0.2463 - sparse_categorical_accuracy: 0.8983 - val_loss: 0.2739 - val_sparse_categorical_accuracy: 0.8849 - learning_rate: 0.0010
    ## Epoch 37/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2403 - sparse_categorical_accuracy: 0.9007 - val_loss: 0.2754 - val_sparse_categorical_accuracy: 0.8821 - learning_rate: 0.0010
    ## Epoch 38/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2417 - sparse_categorical_accuracy: 0.9038 - val_loss: 0.2464 - val_sparse_categorical_accuracy: 0.8974 - learning_rate: 0.0010
    ## Epoch 39/500
    ## 90/90 - 0s - 1ms/step - loss: 0.2407 - sparse_categorical_accuracy: 0.9024 - val_loss: 0.5480 - val_sparse_categorical_accuracy: 0.7517 - learning_rate: 0.0010
    ## Epoch 40/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2379 - sparse_categorical_accuracy: 0.9024 - val_loss: 0.9463 - val_sparse_categorical_accuracy: 0.6477 - learning_rate: 0.0010
    ## Epoch 41/500
    ## 90/90 - 0s - 1ms/step - loss: 0.2271 - sparse_categorical_accuracy: 0.9111 - val_loss: 0.3108 - val_sparse_categorical_accuracy: 0.8627 - learning_rate: 0.0010
    ## Epoch 42/500
    ## 90/90 - 0s - 1ms/step - loss: 0.2142 - sparse_categorical_accuracy: 0.9181 - val_loss: 0.3145 - val_sparse_categorical_accuracy: 0.8682 - learning_rate: 0.0010
    ## Epoch 43/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2221 - sparse_categorical_accuracy: 0.9073 - val_loss: 0.4823 - val_sparse_categorical_accuracy: 0.7850 - learning_rate: 0.0010
    ## Epoch 44/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2202 - sparse_categorical_accuracy: 0.9135 - val_loss: 0.3254 - val_sparse_categorical_accuracy: 0.8502 - learning_rate: 0.0010
    ## Epoch 45/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2196 - sparse_categorical_accuracy: 0.9142 - val_loss: 1.5993 - val_sparse_categorical_accuracy: 0.6172 - learning_rate: 0.0010
    ## Epoch 46/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2180 - sparse_categorical_accuracy: 0.9149 - val_loss: 0.2436 - val_sparse_categorical_accuracy: 0.8890 - learning_rate: 0.0010
    ## Epoch 47/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1999 - sparse_categorical_accuracy: 0.9271 - val_loss: 0.3012 - val_sparse_categorical_accuracy: 0.8682 - learning_rate: 0.0010
    ## Epoch 48/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2032 - sparse_categorical_accuracy: 0.9177 - val_loss: 0.3066 - val_sparse_categorical_accuracy: 0.8488 - learning_rate: 0.0010
    ## Epoch 49/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1894 - sparse_categorical_accuracy: 0.9233 - val_loss: 0.2238 - val_sparse_categorical_accuracy: 0.9098 - learning_rate: 0.0010
    ## Epoch 50/500
    ## 90/90 - 0s - 2ms/step - loss: 0.2086 - sparse_categorical_accuracy: 0.9191 - val_loss: 0.2838 - val_sparse_categorical_accuracy: 0.8918 - learning_rate: 0.0010
    ## Epoch 51/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1863 - sparse_categorical_accuracy: 0.9330 - val_loss: 0.2303 - val_sparse_categorical_accuracy: 0.9029 - learning_rate: 0.0010
    ## Epoch 52/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1740 - sparse_categorical_accuracy: 0.9385 - val_loss: 0.3756 - val_sparse_categorical_accuracy: 0.8655 - learning_rate: 0.0010
    ## Epoch 53/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1662 - sparse_categorical_accuracy: 0.9365 - val_loss: 0.3462 - val_sparse_categorical_accuracy: 0.8294 - learning_rate: 0.0010
    ## Epoch 54/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1574 - sparse_categorical_accuracy: 0.9451 - val_loss: 0.5578 - val_sparse_categorical_accuracy: 0.7642 - learning_rate: 0.0010
    ## Epoch 55/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1467 - sparse_categorical_accuracy: 0.9524 - val_loss: 0.4782 - val_sparse_categorical_accuracy: 0.8003 - learning_rate: 0.0010
    ## Epoch 56/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1464 - sparse_categorical_accuracy: 0.9510 - val_loss: 0.2423 - val_sparse_categorical_accuracy: 0.9071 - learning_rate: 0.0010
    ## Epoch 57/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1370 - sparse_categorical_accuracy: 0.9583 - val_loss: 0.1818 - val_sparse_categorical_accuracy: 0.9154 - learning_rate: 0.0010
    ## Epoch 58/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1258 - sparse_categorical_accuracy: 0.9604 - val_loss: 0.2072 - val_sparse_categorical_accuracy: 0.9265 - learning_rate: 0.0010
    ## Epoch 59/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1470 - sparse_categorical_accuracy: 0.9462 - val_loss: 0.1818 - val_sparse_categorical_accuracy: 0.9196 - learning_rate: 0.0010
    ## Epoch 60/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1315 - sparse_categorical_accuracy: 0.9563 - val_loss: 0.2208 - val_sparse_categorical_accuracy: 0.9001 - learning_rate: 0.0010
    ## Epoch 61/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1167 - sparse_categorical_accuracy: 0.9635 - val_loss: 0.1562 - val_sparse_categorical_accuracy: 0.9320 - learning_rate: 0.0010
    ## Epoch 62/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1348 - sparse_categorical_accuracy: 0.9542 - val_loss: 0.3085 - val_sparse_categorical_accuracy: 0.8558 - learning_rate: 0.0010
    ## Epoch 63/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1274 - sparse_categorical_accuracy: 0.9538 - val_loss: 1.7598 - val_sparse_categorical_accuracy: 0.6269 - learning_rate: 0.0010
    ## Epoch 64/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1209 - sparse_categorical_accuracy: 0.9615 - val_loss: 0.2698 - val_sparse_categorical_accuracy: 0.8738 - learning_rate: 0.0010
    ## Epoch 65/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1172 - sparse_categorical_accuracy: 0.9615 - val_loss: 0.2389 - val_sparse_categorical_accuracy: 0.9057 - learning_rate: 0.0010
    ## Epoch 66/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1096 - sparse_categorical_accuracy: 0.9670 - val_loss: 0.2290 - val_sparse_categorical_accuracy: 0.9057 - learning_rate: 0.0010
    ## Epoch 67/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1192 - sparse_categorical_accuracy: 0.9628 - val_loss: 1.3580 - val_sparse_categorical_accuracy: 0.6852 - learning_rate: 0.0010
    ## Epoch 68/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1226 - sparse_categorical_accuracy: 0.9576 - val_loss: 0.2112 - val_sparse_categorical_accuracy: 0.9085 - learning_rate: 0.0010
    ## Epoch 69/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1219 - sparse_categorical_accuracy: 0.9597 - val_loss: 0.1998 - val_sparse_categorical_accuracy: 0.9307 - learning_rate: 0.0010
    ## Epoch 70/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1133 - sparse_categorical_accuracy: 0.9632 - val_loss: 1.2104 - val_sparse_categorical_accuracy: 0.7393 - learning_rate: 0.0010
    ## Epoch 71/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1086 - sparse_categorical_accuracy: 0.9649 - val_loss: 3.7852 - val_sparse_categorical_accuracy: 0.5908 - learning_rate: 0.0010
    ## Epoch 72/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1080 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.2014 - val_sparse_categorical_accuracy: 0.9029 - learning_rate: 0.0010
    ## Epoch 73/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1021 - sparse_categorical_accuracy: 0.9670 - val_loss: 0.2156 - val_sparse_categorical_accuracy: 0.8960 - learning_rate: 0.0010
    ## Epoch 74/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1065 - sparse_categorical_accuracy: 0.9656 - val_loss: 1.0247 - val_sparse_categorical_accuracy: 0.7351 - learning_rate: 0.0010
    ## Epoch 75/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1289 - sparse_categorical_accuracy: 0.9556 - val_loss: 1.1632 - val_sparse_categorical_accuracy: 0.7129 - learning_rate: 0.0010
    ## Epoch 76/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1133 - sparse_categorical_accuracy: 0.9608 - val_loss: 0.2318 - val_sparse_categorical_accuracy: 0.9057 - learning_rate: 0.0010
    ## Epoch 77/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1007 - sparse_categorical_accuracy: 0.9677 - val_loss: 0.1296 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 0.0010
    ## Epoch 78/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1000 - sparse_categorical_accuracy: 0.9708 - val_loss: 0.5377 - val_sparse_categorical_accuracy: 0.7642 - learning_rate: 0.0010
    ## Epoch 79/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1008 - sparse_categorical_accuracy: 0.9674 - val_loss: 0.2489 - val_sparse_categorical_accuracy: 0.8974 - learning_rate: 0.0010
    ## Epoch 80/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1028 - sparse_categorical_accuracy: 0.9649 - val_loss: 0.1783 - val_sparse_categorical_accuracy: 0.9334 - learning_rate: 0.0010
    ## Epoch 81/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1065 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.3968 - val_sparse_categorical_accuracy: 0.8336 - learning_rate: 0.0010
    ## Epoch 82/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0952 - sparse_categorical_accuracy: 0.9684 - val_loss: 0.1315 - val_sparse_categorical_accuracy: 0.9487 - learning_rate: 0.0010
    ## Epoch 83/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1009 - sparse_categorical_accuracy: 0.9684 - val_loss: 0.3119 - val_sparse_categorical_accuracy: 0.8655 - learning_rate: 0.0010
    ## Epoch 84/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1006 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.2430 - val_sparse_categorical_accuracy: 0.8835 - learning_rate: 0.0010
    ## Epoch 85/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1012 - sparse_categorical_accuracy: 0.9670 - val_loss: 0.2036 - val_sparse_categorical_accuracy: 0.9015 - learning_rate: 0.0010
    ## Epoch 86/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1038 - sparse_categorical_accuracy: 0.9628 - val_loss: 0.2467 - val_sparse_categorical_accuracy: 0.8960 - learning_rate: 0.0010
    ## Epoch 87/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1075 - sparse_categorical_accuracy: 0.9642 - val_loss: 0.1566 - val_sparse_categorical_accuracy: 0.9417 - learning_rate: 0.0010
    ## Epoch 88/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0970 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.1981 - val_sparse_categorical_accuracy: 0.9237 - learning_rate: 0.0010
    ## Epoch 89/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1041 - sparse_categorical_accuracy: 0.9653 - val_loss: 0.1270 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 0.0010
    ## Epoch 90/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0962 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.1483 - val_sparse_categorical_accuracy: 0.9404 - learning_rate: 0.0010
    ## Epoch 91/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1055 - sparse_categorical_accuracy: 0.9649 - val_loss: 0.4081 - val_sparse_categorical_accuracy: 0.8363 - learning_rate: 0.0010
    ## Epoch 92/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0998 - sparse_categorical_accuracy: 0.9670 - val_loss: 0.1236 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 0.0010
    ## Epoch 93/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1020 - sparse_categorical_accuracy: 0.9656 - val_loss: 0.3183 - val_sparse_categorical_accuracy: 0.8599 - learning_rate: 0.0010
    ## Epoch 94/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1039 - sparse_categorical_accuracy: 0.9642 - val_loss: 0.6428 - val_sparse_categorical_accuracy: 0.7656 - learning_rate: 0.0010
    ## Epoch 95/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0978 - sparse_categorical_accuracy: 0.9694 - val_loss: 1.0944 - val_sparse_categorical_accuracy: 0.6394 - learning_rate: 0.0010
    ## Epoch 96/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1107 - sparse_categorical_accuracy: 0.9580 - val_loss: 3.4357 - val_sparse_categorical_accuracy: 0.4924 - learning_rate: 0.0010
    ## Epoch 97/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1109 - sparse_categorical_accuracy: 0.9632 - val_loss: 0.2506 - val_sparse_categorical_accuracy: 0.9015 - learning_rate: 0.0010
    ## Epoch 98/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0926 - sparse_categorical_accuracy: 0.9708 - val_loss: 0.1989 - val_sparse_categorical_accuracy: 0.9126 - learning_rate: 0.0010
    ## Epoch 99/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0935 - sparse_categorical_accuracy: 0.9649 - val_loss: 0.4732 - val_sparse_categorical_accuracy: 0.8183 - learning_rate: 0.0010
    ## Epoch 100/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0955 - sparse_categorical_accuracy: 0.9674 - val_loss: 0.1228 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 0.0010
    ## Epoch 101/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0873 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.1301 - val_sparse_categorical_accuracy: 0.9487 - learning_rate: 0.0010
    ## Epoch 102/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0966 - sparse_categorical_accuracy: 0.9694 - val_loss: 0.2766 - val_sparse_categorical_accuracy: 0.8724 - learning_rate: 0.0010
    ## Epoch 103/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0922 - sparse_categorical_accuracy: 0.9694 - val_loss: 0.2188 - val_sparse_categorical_accuracy: 0.9154 - learning_rate: 0.0010
    ## Epoch 104/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1009 - sparse_categorical_accuracy: 0.9635 - val_loss: 0.2078 - val_sparse_categorical_accuracy: 0.9209 - learning_rate: 0.0010
    ## Epoch 105/500
    ## 90/90 - 0s - 1ms/step - loss: 0.1027 - sparse_categorical_accuracy: 0.9639 - val_loss: 0.9116 - val_sparse_categorical_accuracy: 0.7559 - learning_rate: 0.0010
    ## Epoch 106/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0930 - sparse_categorical_accuracy: 0.9691 - val_loss: 1.0226 - val_sparse_categorical_accuracy: 0.7046 - learning_rate: 0.0010
    ## Epoch 107/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0886 - sparse_categorical_accuracy: 0.9694 - val_loss: 0.9237 - val_sparse_categorical_accuracy: 0.7143 - learning_rate: 0.0010
    ## Epoch 108/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0939 - sparse_categorical_accuracy: 0.9698 - val_loss: 0.1203 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 0.0010
    ## Epoch 109/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0949 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.1665 - val_sparse_categorical_accuracy: 0.9417 - learning_rate: 0.0010
    ## Epoch 110/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0997 - sparse_categorical_accuracy: 0.9674 - val_loss: 0.5271 - val_sparse_categorical_accuracy: 0.7906 - learning_rate: 0.0010
    ## Epoch 111/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0874 - sparse_categorical_accuracy: 0.9698 - val_loss: 0.1229 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 0.0010
    ## Epoch 112/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0918 - sparse_categorical_accuracy: 0.9674 - val_loss: 0.6369 - val_sparse_categorical_accuracy: 0.7947 - learning_rate: 0.0010
    ## Epoch 113/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0897 - sparse_categorical_accuracy: 0.9681 - val_loss: 2.3587 - val_sparse_categorical_accuracy: 0.6449 - learning_rate: 0.0010
    ## Epoch 114/500
    ## 90/90 - 0s - 2ms/step - loss: 0.1061 - sparse_categorical_accuracy: 0.9660 - val_loss: 0.2340 - val_sparse_categorical_accuracy: 0.9112 - learning_rate: 0.0010
    ## Epoch 115/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0877 - sparse_categorical_accuracy: 0.9708 - val_loss: 0.1491 - val_sparse_categorical_accuracy: 0.9404 - learning_rate: 0.0010
    ## Epoch 116/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0932 - sparse_categorical_accuracy: 0.9733 - val_loss: 1.1600 - val_sparse_categorical_accuracy: 0.7240 - learning_rate: 0.0010
    ## Epoch 117/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0814 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.3382 - val_sparse_categorical_accuracy: 0.8890 - learning_rate: 0.0010
    ## Epoch 118/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0900 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.1797 - val_sparse_categorical_accuracy: 0.9223 - learning_rate: 0.0010
    ## Epoch 119/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0873 - sparse_categorical_accuracy: 0.9701 - val_loss: 0.1383 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 0.0010
    ## Epoch 120/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0877 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.1644 - val_sparse_categorical_accuracy: 0.9445 - learning_rate: 0.0010
    ## Epoch 121/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0888 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.2866 - val_sparse_categorical_accuracy: 0.8918 - learning_rate: 0.0010
    ## Epoch 122/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0932 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.6170 - val_sparse_categorical_accuracy: 0.8003 - learning_rate: 0.0010
    ## Epoch 123/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0914 - sparse_categorical_accuracy: 0.9708 - val_loss: 1.4002 - val_sparse_categorical_accuracy: 0.7101 - learning_rate: 0.0010
    ## Epoch 124/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0919 - sparse_categorical_accuracy: 0.9670 - val_loss: 0.4126 - val_sparse_categorical_accuracy: 0.8239 - learning_rate: 0.0010
    ## Epoch 125/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0929 - sparse_categorical_accuracy: 0.9677 - val_loss: 0.1481 - val_sparse_categorical_accuracy: 0.9376 - learning_rate: 0.0010
    ## Epoch 126/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0807 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.3654 - val_sparse_categorical_accuracy: 0.8613 - learning_rate: 0.0010
    ## Epoch 127/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0871 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.1755 - val_sparse_categorical_accuracy: 0.9293 - learning_rate: 0.0010
    ## Epoch 128/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0906 - sparse_categorical_accuracy: 0.9691 - val_loss: 0.2267 - val_sparse_categorical_accuracy: 0.9098 - learning_rate: 0.0010
    ## Epoch 129/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0812 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.2041 - val_sparse_categorical_accuracy: 0.9112 - learning_rate: 5.0000e-04
    ## Epoch 130/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0872 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.1053 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 5.0000e-04
    ## Epoch 131/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0711 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.1209 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 5.0000e-04
    ## Epoch 132/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0704 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.1679 - val_sparse_categorical_accuracy: 0.9307 - learning_rate: 5.0000e-04
    ## Epoch 133/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0657 - sparse_categorical_accuracy: 0.9813 - val_loss: 0.2621 - val_sparse_categorical_accuracy: 0.9057 - learning_rate: 5.0000e-04
    ## Epoch 134/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0754 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.1166 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 5.0000e-04
    ## Epoch 135/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0747 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.1042 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 5.0000e-04
    ## Epoch 136/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0764 - sparse_categorical_accuracy: 0.9719 - val_loss: 0.3386 - val_sparse_categorical_accuracy: 0.8779 - learning_rate: 5.0000e-04
    ## Epoch 137/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0789 - sparse_categorical_accuracy: 0.9729 - val_loss: 0.1587 - val_sparse_categorical_accuracy: 0.9445 - learning_rate: 5.0000e-04
    ## Epoch 138/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0729 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.1540 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 5.0000e-04
    ## Epoch 139/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0796 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.1252 - val_sparse_categorical_accuracy: 0.9501 - learning_rate: 5.0000e-04
    ## Epoch 140/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0681 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.1587 - val_sparse_categorical_accuracy: 0.9445 - learning_rate: 5.0000e-04
    ## Epoch 141/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0739 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.1396 - val_sparse_categorical_accuracy: 0.9376 - learning_rate: 5.0000e-04
    ## Epoch 142/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0704 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.2482 - val_sparse_categorical_accuracy: 0.8932 - learning_rate: 5.0000e-04
    ## Epoch 143/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0754 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.2152 - val_sparse_categorical_accuracy: 0.8974 - learning_rate: 5.0000e-04
    ## Epoch 144/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0675 - sparse_categorical_accuracy: 0.9795 - val_loss: 0.2873 - val_sparse_categorical_accuracy: 0.8904 - learning_rate: 5.0000e-04
    ## Epoch 145/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0689 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.1540 - val_sparse_categorical_accuracy: 0.9459 - learning_rate: 5.0000e-04
    ## Epoch 146/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0703 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.1351 - val_sparse_categorical_accuracy: 0.9473 - learning_rate: 5.0000e-04
    ## Epoch 147/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0740 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.1116 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 5.0000e-04
    ## Epoch 148/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0758 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.1083 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 5.0000e-04
    ## Epoch 149/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0718 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.1052 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 5.0000e-04
    ## Epoch 150/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0715 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.5035 - val_sparse_categorical_accuracy: 0.8433 - learning_rate: 5.0000e-04
    ## Epoch 151/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0728 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.1050 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 5.0000e-04
    ## Epoch 152/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0716 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.3563 - val_sparse_categorical_accuracy: 0.8752 - learning_rate: 5.0000e-04
    ## Epoch 153/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0677 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.2917 - val_sparse_categorical_accuracy: 0.8946 - learning_rate: 5.0000e-04
    ## Epoch 154/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0673 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.1561 - val_sparse_categorical_accuracy: 0.9320 - learning_rate: 5.0000e-04
    ## Epoch 155/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0705 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.1184 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 5.0000e-04
    ## Epoch 156/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0703 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.0999 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 2.5000e-04
    ## Epoch 157/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0664 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.1080 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 2.5000e-04
    ## Epoch 158/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0660 - sparse_categorical_accuracy: 0.9795 - val_loss: 0.1113 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 2.5000e-04
    ## Epoch 159/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0644 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.1069 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 2.5000e-04
    ## Epoch 160/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0663 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.1082 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 2.5000e-04
    ## Epoch 161/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0642 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1065 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 2.5000e-04
    ## Epoch 162/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0594 - sparse_categorical_accuracy: 0.9813 - val_loss: 0.0989 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 2.5000e-04
    ## Epoch 163/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0573 - sparse_categorical_accuracy: 0.9813 - val_loss: 0.1050 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 2.5000e-04
    ## Epoch 164/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0604 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.1254 - val_sparse_categorical_accuracy: 0.9473 - learning_rate: 2.5000e-04
    ## Epoch 165/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0579 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.1167 - val_sparse_categorical_accuracy: 0.9473 - learning_rate: 2.5000e-04
    ## Epoch 166/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0580 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1052 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 2.5000e-04
    ## Epoch 167/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0585 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.1125 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 2.5000e-04
    ## Epoch 168/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0696 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.1026 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 2.5000e-04
    ## Epoch 169/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0612 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.1050 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 2.5000e-04
    ## Epoch 170/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0614 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.1449 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 2.5000e-04
    ## Epoch 171/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0568 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.2052 - val_sparse_categorical_accuracy: 0.9265 - learning_rate: 2.5000e-04
    ## Epoch 172/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0599 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1332 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 2.5000e-04
    ## Epoch 173/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0589 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.1152 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 2.5000e-04
    ## Epoch 174/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0602 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.1074 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 2.5000e-04
    ## Epoch 175/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0608 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.1109 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
    ## Epoch 176/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0581 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.0976 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 2.5000e-04
    ## Epoch 177/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0575 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.1137 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 2.5000e-04
    ## Epoch 178/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0560 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.1744 - val_sparse_categorical_accuracy: 0.9237 - learning_rate: 2.5000e-04
    ## Epoch 179/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0557 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1091 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 2.5000e-04
    ## Epoch 180/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0573 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.1373 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 2.5000e-04
    ## Epoch 181/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0601 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.1057 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 2.5000e-04
    ## Epoch 182/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0555 - sparse_categorical_accuracy: 0.9813 - val_loss: 0.1035 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 2.5000e-04
    ## Epoch 183/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0587 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.1047 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 2.5000e-04
    ## Epoch 184/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0545 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.1190 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 2.5000e-04
    ## Epoch 185/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0557 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.1244 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 2.5000e-04
    ## Epoch 186/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0628 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.1057 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 2.5000e-04
    ## Epoch 187/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0566 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.1028 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 2.5000e-04
    ## Epoch 188/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0563 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.2356 - val_sparse_categorical_accuracy: 0.9112 - learning_rate: 2.5000e-04
    ## Epoch 189/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0520 - sparse_categorical_accuracy: 0.9872 - val_loss: 0.2162 - val_sparse_categorical_accuracy: 0.9209 - learning_rate: 2.5000e-04
    ## Epoch 190/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0647 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.1439 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 2.5000e-04
    ## Epoch 191/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0557 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.0982 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 2.5000e-04
    ## Epoch 192/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0605 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.1251 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 2.5000e-04
    ## Epoch 193/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0576 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.1062 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 2.5000e-04
    ## Epoch 194/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0563 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.4670 - val_sparse_categorical_accuracy: 0.8169 - learning_rate: 2.5000e-04
    ## Epoch 195/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0555 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.1543 - val_sparse_categorical_accuracy: 0.9334 - learning_rate: 2.5000e-04
    ## Epoch 196/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0576 - sparse_categorical_accuracy: 0.9813 - val_loss: 0.0970 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 2.5000e-04
    ## Epoch 197/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0553 - sparse_categorical_accuracy: 0.9813 - val_loss: 0.1862 - val_sparse_categorical_accuracy: 0.9320 - learning_rate: 2.5000e-04
    ## Epoch 198/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0537 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.1012 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 2.5000e-04
    ## Epoch 199/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0554 - sparse_categorical_accuracy: 0.9813 - val_loss: 0.1001 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 2.5000e-04
    ## Epoch 200/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0569 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1632 - val_sparse_categorical_accuracy: 0.9404 - learning_rate: 2.5000e-04
    ## Epoch 201/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0551 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.1066 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 2.5000e-04
    ## Epoch 202/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0552 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.1800 - val_sparse_categorical_accuracy: 0.9334 - learning_rate: 2.5000e-04
    ## Epoch 203/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0535 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.1152 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 2.5000e-04
    ## Epoch 204/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0502 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.1524 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 2.5000e-04
    ## Epoch 205/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0544 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.1535 - val_sparse_categorical_accuracy: 0.9362 - learning_rate: 2.5000e-04
    ## Epoch 206/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0540 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.1371 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 2.5000e-04
    ## Epoch 207/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0589 - sparse_categorical_accuracy: 0.9795 - val_loss: 0.3198 - val_sparse_categorical_accuracy: 0.8863 - learning_rate: 2.5000e-04
    ## Epoch 208/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0564 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.2081 - val_sparse_categorical_accuracy: 0.9251 - learning_rate: 2.5000e-04
    ## Epoch 209/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0520 - sparse_categorical_accuracy: 0.9813 - val_loss: 0.1350 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 2.5000e-04
    ## Epoch 210/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0573 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.1064 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 2.5000e-04
    ## Epoch 211/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0548 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.2211 - val_sparse_categorical_accuracy: 0.9140 - learning_rate: 2.5000e-04
    ## Epoch 212/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0526 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.1249 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 2.5000e-04
    ## Epoch 213/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0563 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.2032 - val_sparse_categorical_accuracy: 0.9209 - learning_rate: 2.5000e-04
    ## Epoch 214/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0548 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.1827 - val_sparse_categorical_accuracy: 0.9334 - learning_rate: 2.5000e-04
    ## Epoch 215/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0520 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.2326 - val_sparse_categorical_accuracy: 0.9154 - learning_rate: 2.5000e-04
    ## Epoch 216/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0531 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.1158 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 2.5000e-04
    ## Epoch 217/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0503 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.1009 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 1.2500e-04
    ## Epoch 218/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0522 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.1010 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 1.2500e-04
    ## Epoch 219/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0489 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.1585 - val_sparse_categorical_accuracy: 0.9487 - learning_rate: 1.2500e-04
    ## Epoch 220/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0475 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.1311 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 1.2500e-04
    ## Epoch 221/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0494 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.1123 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.2500e-04
    ## Epoch 222/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0512 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.1083 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.2500e-04
    ## Epoch 223/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0492 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.1226 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 1.2500e-04
    ## Epoch 224/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0465 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0984 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.2500e-04
    ## Epoch 225/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0511 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.1238 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.2500e-04
    ## Epoch 226/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0467 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.1083 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.2500e-04
    ## Epoch 227/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0518 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0971 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.2500e-04
    ## Epoch 228/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0465 - sparse_categorical_accuracy: 0.9875 - val_loss: 0.1080 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 1.2500e-04
    ## Epoch 229/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0477 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0977 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
    ## Epoch 230/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0465 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0970 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.2500e-04
    ## Epoch 231/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0513 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0973 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 1.2500e-04
    ## Epoch 232/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0468 - sparse_categorical_accuracy: 0.9872 - val_loss: 0.0987 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.2500e-04
    ## Epoch 233/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0480 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0963 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 1.2500e-04
    ## Epoch 234/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0487 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.1033 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
    ## Epoch 235/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0490 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.1129 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 1.2500e-04
    ## Epoch 236/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0494 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.1230 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.2500e-04
    ## Epoch 237/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0483 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.1283 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 1.2500e-04
    ## Epoch 238/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0476 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0988 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.2500e-04
    ## Epoch 239/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0476 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0959 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 1.2500e-04
    ## Epoch 240/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0466 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0974 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 1.2500e-04
    ## Epoch 241/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0494 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.1366 - val_sparse_categorical_accuracy: 0.9501 - learning_rate: 1.2500e-04
    ## Epoch 242/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0477 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.1099 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.2500e-04
    ## Epoch 243/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0511 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.1000 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 1.2500e-04
    ## Epoch 244/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0513 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1100 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 1.2500e-04
    ## Epoch 245/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0482 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.1016 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.2500e-04
    ## Epoch 246/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0493 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.1089 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.2500e-04
    ## Epoch 247/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0450 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.1003 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.2500e-04
    ## Epoch 248/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0442 - sparse_categorical_accuracy: 0.9878 - val_loss: 0.0991 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.2500e-04
    ## Epoch 249/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0487 - sparse_categorical_accuracy: 0.9882 - val_loss: 0.1016 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 1.2500e-04
    ## Epoch 250/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0489 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.1236 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 1.2500e-04
    ## Epoch 251/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0479 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.1039 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 1.2500e-04
    ## Epoch 252/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0503 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.1209 - val_sparse_categorical_accuracy: 0.9487 - learning_rate: 1.2500e-04
    ## Epoch 253/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0486 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0988 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 1.2500e-04
    ## Epoch 254/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0500 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.1133 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 1.2500e-04
    ## Epoch 255/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0459 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.1025 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 1.2500e-04
    ## Epoch 256/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0476 - sparse_categorical_accuracy: 0.9882 - val_loss: 0.0999 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.2500e-04
    ## Epoch 257/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0471 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.1021 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
    ## Epoch 258/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0440 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.1016 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 1.2500e-04
    ## Epoch 259/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0452 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.1037 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 1.2500e-04
    ## Epoch 260/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0445 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.1032 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 1.0000e-04
    ## Epoch 261/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0480 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0980 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.0000e-04
    ## Epoch 262/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0444 - sparse_categorical_accuracy: 0.9875 - val_loss: 0.1075 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 1.0000e-04
    ## Epoch 263/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0439 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.1209 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 1.0000e-04
    ## Epoch 264/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0473 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.1056 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
    ## Epoch 265/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0509 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.0946 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.0000e-04
    ## Epoch 266/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0462 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0962 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.0000e-04
    ## Epoch 267/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0439 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0964 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
    ## Epoch 268/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0426 - sparse_categorical_accuracy: 0.9896 - val_loss: 0.1286 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 1.0000e-04
    ## Epoch 269/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0472 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0975 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 1.0000e-04
    ## Epoch 270/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0407 - sparse_categorical_accuracy: 0.9910 - val_loss: 0.0986 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 1.0000e-04
    ## Epoch 271/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0442 - sparse_categorical_accuracy: 0.9882 - val_loss: 0.0982 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.0000e-04
    ## Epoch 272/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0458 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.1031 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
    ## Epoch 273/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0460 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.1008 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 1.0000e-04
    ## Epoch 274/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0444 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.1200 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.0000e-04
    ## Epoch 275/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0423 - sparse_categorical_accuracy: 0.9878 - val_loss: 0.1213 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.0000e-04
    ## Epoch 276/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0447 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0980 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.0000e-04
    ## Epoch 277/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0441 - sparse_categorical_accuracy: 0.9882 - val_loss: 0.0982 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
    ## Epoch 278/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0447 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.1105 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
    ## Epoch 279/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0454 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.1059 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 1.0000e-04
    ## Epoch 280/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0449 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.1005 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.0000e-04
    ## Epoch 281/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0425 - sparse_categorical_accuracy: 0.9892 - val_loss: 0.1093 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.0000e-04
    ## Epoch 282/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0417 - sparse_categorical_accuracy: 0.9885 - val_loss: 0.0977 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 1.0000e-04
    ## Epoch 283/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0464 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.1028 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 1.0000e-04
    ## Epoch 284/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0472 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.1080 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.0000e-04
    ## Epoch 285/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0444 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.1037 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.0000e-04
    ## Epoch 286/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0441 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.1106 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
    ## Epoch 287/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0424 - sparse_categorical_accuracy: 0.9875 - val_loss: 0.0986 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 1.0000e-04
    ## Epoch 288/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0433 - sparse_categorical_accuracy: 0.9885 - val_loss: 0.1010 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.0000e-04
    ## Epoch 289/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0437 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.1025 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.0000e-04
    ## Epoch 290/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0453 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0979 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.0000e-04
    ## Epoch 291/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0433 - sparse_categorical_accuracy: 0.9875 - val_loss: 0.0983 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.0000e-04
    ## Epoch 292/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0490 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.1181 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.0000e-04
    ## Epoch 293/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0474 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.1126 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
    ## Epoch 294/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0441 - sparse_categorical_accuracy: 0.9872 - val_loss: 0.1398 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 1.0000e-04
    ## Epoch 295/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0476 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0982 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.0000e-04
    ## Epoch 296/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0490 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0995 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 1.0000e-04
    ## Epoch 297/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0476 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.1214 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.0000e-04
    ## Epoch 298/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0449 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0981 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 1.0000e-04
    ## Epoch 299/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0405 - sparse_categorical_accuracy: 0.9885 - val_loss: 0.1292 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.0000e-04
    ## Epoch 300/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0419 - sparse_categorical_accuracy: 0.9875 - val_loss: 0.0978 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 1.0000e-04
    ## Epoch 301/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0451 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0989 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.0000e-04
    ## Epoch 302/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0404 - sparse_categorical_accuracy: 0.9896 - val_loss: 0.1040 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
    ## Epoch 303/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0411 - sparse_categorical_accuracy: 0.9899 - val_loss: 0.0988 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.0000e-04
    ## Epoch 304/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0448 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.1179 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
    ## Epoch 305/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0448 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.1219 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
    ## Epoch 306/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0396 - sparse_categorical_accuracy: 0.9892 - val_loss: 0.1043 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.0000e-04
    ## Epoch 307/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0459 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.1082 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 1.0000e-04
    ## Epoch 308/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0451 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0968 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.0000e-04
    ## Epoch 309/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0448 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0997 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 1.0000e-04
    ## Epoch 310/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0425 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0988 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.0000e-04
    ## Epoch 311/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0433 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.1013 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 1.0000e-04
    ## Epoch 312/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0440 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.1103 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.0000e-04
    ## Epoch 313/500
    ## 90/90 - 0s - 1ms/step - loss: 0.0411 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.1058 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.0000e-04
    ## Epoch 314/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0425 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.1183 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.0000e-04
    ## Epoch 315/500
    ## 90/90 - 0s - 2ms/step - loss: 0.0462 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0987 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.0000e-04
    ## Epoch 315: early stopping

## Evaluate model on test data

``` r
model <- load_model("best_model.keras")

results <- model |> evaluate(x_test, y_test)
```

    ## 42/42 - 1s - 15ms/step - loss: 0.0915 - sparse_categorical_accuracy: 0.9727

``` r
str(results)
```

    ## List of 2
    ##  $ loss                       : num 0.0915
    ##  $ sparse_categorical_accuracy: num 0.973

``` r
cat(
  "Test accuracy: ", results$sparse_categorical_accuracy, "\n",
  "Test loss: ", results$loss, "\n",
  sep = ""
)
```

    ## Test accuracy: 0.9727273
    ## Test loss: 0.09153023

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
