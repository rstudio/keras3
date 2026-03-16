# Simple MNIST convnet

## Setup

``` r
library(keras3)
```

## Prepare the data

``` r
# Model / data parameters
num_classes <- 10
input_shape <- c(28, 28, 1)

# Load the data and split it between train and test sets
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_mnist()

# Scale images to the [0, 1] range
x_train <- x_train / 255
x_test <- x_test / 255
# Make sure images have shape (28, 28, 1)
x_train <- op_expand_dims(x_train, -1)
x_test <- op_expand_dims(x_test, -1)


dim(x_train)
```

    ## [1] 60000    28    28     1

``` r
dim(x_test)
```

    ## [1] 10000    28    28     1

``` r
# convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)
```

## Build the model

``` r
model <- keras_model_sequential(input_shape = input_shape)
model |>
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") |>
  layer_max_pooling_2d(pool_size = c(2, 2)) |>
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") |>
  layer_max_pooling_2d(pool_size = c(2, 2)) |>
  layer_flatten() |>
  layer_dropout(rate = 0.5) |>
  layer_dense(units = num_classes, activation = "softmax")

summary(model)
```

    ## Model: "sequential"
    ## ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ## ┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
    ## ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    ## │ conv2d (Conv2D)                 │ (None, 26, 26, 32)     │           320 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ max_pooling2d (MaxPooling2D)    │ (None, 13, 13, 32)     │             0 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ conv2d_1 (Conv2D)               │ (None, 11, 11, 64)     │        18,496 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ max_pooling2d_1 (MaxPooling2D)  │ (None, 5, 5, 64)       │             0 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ flatten (Flatten)               │ (None, 1600)           │             0 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ dropout (Dropout)               │ (None, 1600)           │             0 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ dense (Dense)                   │ (None, 10)             │        16,010 │
    ## └─────────────────────────────────┴────────────────────────┴───────────────┘
    ##  Total params: 34,826 (136.04 KB)
    ##  Trainable params: 34,826 (136.04 KB)
    ##  Non-trainable params: 0 (0.00 B)

## Train the model

``` r
batch_size <- 128
epochs <- 15

model |> compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

model |> fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.1
)
```

    ## Epoch 1/15
    ## 422/422 - 5s - 11ms/step - accuracy: 0.8896 - loss: 0.3635 - val_accuracy: 0.9785 - val_loss: 0.0790
    ## Epoch 2/15
    ## 422/422 - 1s - 2ms/step - accuracy: 0.9666 - loss: 0.1110 - val_accuracy: 0.9852 - val_loss: 0.0549
    ## Epoch 3/15
    ## 422/422 - 1s - 2ms/step - accuracy: 0.9743 - loss: 0.0823 - val_accuracy: 0.9882 - val_loss: 0.0442
    ## Epoch 4/15
    ## 422/422 - 1s - 2ms/step - accuracy: 0.9786 - loss: 0.0693 - val_accuracy: 0.9895 - val_loss: 0.0399
    ## Epoch 5/15
    ## 422/422 - 1s - 2ms/step - accuracy: 0.9804 - loss: 0.0625 - val_accuracy: 0.9900 - val_loss: 0.0353
    ## Epoch 6/15
    ## 422/422 - 1s - 2ms/step - accuracy: 0.9824 - loss: 0.0558 - val_accuracy: 0.9912 - val_loss: 0.0331
    ## Epoch 7/15
    ## 422/422 - 1s - 2ms/step - accuracy: 0.9836 - loss: 0.0501 - val_accuracy: 0.9917 - val_loss: 0.0311
    ## Epoch 8/15
    ## 422/422 - 1s - 2ms/step - accuracy: 0.9852 - loss: 0.0480 - val_accuracy: 0.9917 - val_loss: 0.0310
    ## Epoch 9/15
    ## 422/422 - 1s - 2ms/step - accuracy: 0.9861 - loss: 0.0445 - val_accuracy: 0.9918 - val_loss: 0.0300
    ## Epoch 10/15
    ## 422/422 - 1s - 2ms/step - accuracy: 0.9863 - loss: 0.0440 - val_accuracy: 0.9912 - val_loss: 0.0295
    ## Epoch 11/15
    ## 422/422 - 1s - 2ms/step - accuracy: 0.9870 - loss: 0.0393 - val_accuracy: 0.9917 - val_loss: 0.0307
    ## Epoch 12/15
    ## 422/422 - 1s - 2ms/step - accuracy: 0.9876 - loss: 0.0373 - val_accuracy: 0.9922 - val_loss: 0.0291
    ## Epoch 13/15
    ## 422/422 - 1s - 2ms/step - accuracy: 0.9887 - loss: 0.0348 - val_accuracy: 0.9920 - val_loss: 0.0292
    ## Epoch 14/15
    ## 422/422 - 1s - 2ms/step - accuracy: 0.9890 - loss: 0.0342 - val_accuracy: 0.9928 - val_loss: 0.0274
    ## Epoch 15/15
    ## 422/422 - 1s - 2ms/step - accuracy: 0.9897 - loss: 0.0322 - val_accuracy: 0.9925 - val_loss: 0.0274

## Evaluate the trained model

``` r
score <- model |> evaluate(x_test, y_test, verbose = 0)
score
```

    ## $accuracy
    ## [1] 0.9914
    ##
    ## $loss
    ## [1] 0.02436747
