---
title: Train a Siamese MLP on pairs of digits from the MNIST dataset.
author: Mehdi, Ivo Kwee - https://github.com/ivokwee
date-created: 2020/05/30
last-modified: 2020/04/21
domain: vision
category: intermediate
output: rmarkdown::html_vignette
knit: ({source(here::here("tools/knit.R")); knit_vignette})
tether: https://raw.githubusercontent.com/keras-team/keras-io/master/examples/vision/siamese_contrastive.py
---

## Introduction

[Siamese Networks](https://en.wikipedia.org/wiki/Siamese_neural_network)
are neural networks which share weights between two or more sister networks,
each producing embedding vectors of its respective inputs.

In supervised similarity learning, the networks are then trained to maximize the
contrast (distance) between embeddings of inputs of different classes, while minimizing the distance between
embeddings of similar classes, resulting in embedding spaces that reflect
the class segmentation of the training inputs.

This implementation loosely follows Hadsell-et-al.'06 [1] (see paper for mode
details) but the euclidean distance is replaced by a subtraction
layer and one fully-connect (FC) layer.

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
     https://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 98.11% test accuracy after 20 epochs. 3 seconds per epoch
on a AMD Ryzen 7 PRO 4750U (CPU)


``` r
library(keras3)
```



``` r
contrastive_loss <- function(y_true, y_pred) {
    # Contrastive loss from Hadsell-et-al.'06
    # https://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    margin = 1
    margin_square = op_square(op_maximum(margin - (y_pred), 0))
    op_mean((1 - y_true) * op_square(y_pred) + (y_true) * margin_square)
}
```

## Create pairs of images

We will train the model to differentiate between digits of different classes. For
example, digit `0` needs to be differentiated from the rest of the
digits (`1` through `9`), digit `1` - from `0` and `2` through `9`, and so on.
To carry this out, we will select N random images from class A (for example,
for digit `0`) and pair them with N random images from another class B
(for example, for digit `1`). Then, we can repeat this process for all classes
of digits (until digit `9`). Once we have paired digit `0` with other digits,
we can repeat this process for the remaining classes for the rest of the digits
(from `1` until `9`).

``` r
create_pairs <- function(x, y) {
    # Positive and negative pair creation.
    # Alternates between positive and negative pairs.
    digit_indices <- tapply(1:length(y), y, list)
    y1 <- y
    y2 <- sapply(y, function(a) sample(0:9,1,prob=0.1+0.8*(0:9==a)))
    idx1 <- 1:nrow(x)
    idx2 <- sapply(as.character(y2), function(a) sample(digit_indices[[a]],1))
    is_same  <- 1*(y1==y2)
    list(pair1 = x[idx1,], pair2 = x[idx2,], y = is_same)
}

compute_accuracy <- function(predictions, labels) {
    # Compute classification accuracy with a fixed threshold on distances.
    mean(labels[predictions > 0.5])
}
```



``` r
# the data, shuffled and split between train and test sets
mnist   <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test  <- mnist$test$x
y_test  <- mnist$test$y
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test  <- array_reshape(x_test, c(nrow(x_test), 784))
x_train <- x_train / 255
x_test  <- x_test / 255

# create training+test positive and negative pairs
tr <- create_pairs(x_train, y_train)
te <- create_pairs(x_test,  y_test)

names(tr)
```

```
## [1] "pair1" "pair2" "y"
```

## Network definition


``` r
# input layers
input_dim = 784
input_1 <- layer_input(shape = c(input_dim))
input_2 <- layer_input(shape = c(input_dim))

# definition of the base network that will be shared
base_network <- keras_model_sequential() %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 128, activation = 'relu')

# because we re-use the same instance `base_network`, the weights of
# the network will be shared across the two branches
branch_1 <- base_network(input_1)
branch_2 <- base_network(input_2)

# merging layer
out <- layer_subtract(list(branch_1, branch_2)) %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 16, activation = 'relu') %>%
    layer_dense(1, activation = "sigmoid")

# create and compile model
model <- keras_model(list(input_1, input_2), out)
```

## Train


``` r
model %>% compile(
    optimizer = "rmsprop",
    #loss = "binary_crossentropy",
    loss = contrastive_loss,
    metrics = metric_binary_accuracy
)

history <- model %>% fit(
    list(tr$pair1, tr$pair2), tr$y,
    batch_size = 128,
    epochs = 20,
    validation_data = list(
        list(te$pair1, te$pair2),
        te$y
    )
)
```

```
## Epoch 1/20
## 469/469 - 5s - 11ms/step - binary_accuracy: 0.7568 - loss: 0.1640 - val_binary_accuracy: 0.8796 - val_loss: 0.0981
## Epoch 2/20
## 469/469 - 1s - 2ms/step - binary_accuracy: 0.8892 - loss: 0.0870 - val_binary_accuracy: 0.9222 - val_loss: 0.0633
## Epoch 3/20
## 469/469 - 1s - 1ms/step - binary_accuracy: 0.9240 - loss: 0.0598 - val_binary_accuracy: 0.9398 - val_loss: 0.0490
## Epoch 4/20
## 469/469 - 1s - 1ms/step - binary_accuracy: 0.9433 - loss: 0.0452 - val_binary_accuracy: 0.9485 - val_loss: 0.0395
## Epoch 5/20
## 469/469 - 1s - 2ms/step - binary_accuracy: 0.9569 - loss: 0.0349 - val_binary_accuracy: 0.9609 - val_loss: 0.0314
## Epoch 6/20
## 469/469 - 1s - 1ms/step - binary_accuracy: 0.9643 - loss: 0.0287 - val_binary_accuracy: 0.9634 - val_loss: 0.0287
## Epoch 7/20
## 469/469 - 1s - 2ms/step - binary_accuracy: 0.9699 - loss: 0.0243 - val_binary_accuracy: 0.9652 - val_loss: 0.0276
## Epoch 8/20
## 469/469 - 1s - 2ms/step - binary_accuracy: 0.9745 - loss: 0.0210 - val_binary_accuracy: 0.9680 - val_loss: 0.0261
## Epoch 9/20
## 469/469 - 1s - 1ms/step - binary_accuracy: 0.9769 - loss: 0.0190 - val_binary_accuracy: 0.9681 - val_loss: 0.0260
## Epoch 10/20
## 469/469 - 1s - 1ms/step - binary_accuracy: 0.9785 - loss: 0.0174 - val_binary_accuracy: 0.9706 - val_loss: 0.0236
## Epoch 11/20
## 469/469 - 1s - 2ms/step - binary_accuracy: 0.9811 - loss: 0.0153 - val_binary_accuracy: 0.9722 - val_loss: 0.0229
## Epoch 12/20
## 469/469 - 1s - 1ms/step - binary_accuracy: 0.9824 - loss: 0.0144 - val_binary_accuracy: 0.9723 - val_loss: 0.0220
## Epoch 13/20
## 469/469 - 1s - 2ms/step - binary_accuracy: 0.9832 - loss: 0.0135 - val_binary_accuracy: 0.9739 - val_loss: 0.0212
## Epoch 14/20
## 469/469 - 1s - 1ms/step - binary_accuracy: 0.9845 - loss: 0.0127 - val_binary_accuracy: 0.9753 - val_loss: 0.0204
## Epoch 15/20
## 469/469 - 1s - 2ms/step - binary_accuracy: 0.9860 - loss: 0.0117 - val_binary_accuracy: 0.9732 - val_loss: 0.0209
## Epoch 16/20
## 469/469 - 1s - 2ms/step - binary_accuracy: 0.9871 - loss: 0.0108 - val_binary_accuracy: 0.9763 - val_loss: 0.0197
## Epoch 17/20
## 469/469 - 1s - 1ms/step - binary_accuracy: 0.9871 - loss: 0.0105 - val_binary_accuracy: 0.9737 - val_loss: 0.0203
## Epoch 18/20
## 469/469 - 1s - 1ms/step - binary_accuracy: 0.9877 - loss: 0.0101 - val_binary_accuracy: 0.9742 - val_loss: 0.0208
## Epoch 19/20
## 469/469 - 1s - 2ms/step - binary_accuracy: 0.9881 - loss: 0.0096 - val_binary_accuracy: 0.9765 - val_loss: 0.0201
## Epoch 20/20
## 469/469 - 1s - 1ms/step - binary_accuracy: 0.9895 - loss: 0.0086 - val_binary_accuracy: 0.9746 - val_loss: 0.0202
```

``` r
plot(history)
```

![plot of chunk unnamed-chunk-5](mnist_siamese_graph/unnamed-chunk-5-1.png)

## Evaluate


``` r
# compute final accuracy on training and test sets

tr_pred <- predict(model, list(tr$pair1, tr$pair2))[,1]
```

```
## 1875/1875 - 2s - 845us/step
```

``` r
tr_acc  <- compute_accuracy(tr_pred, tr$y)
te_pred <- predict(model, list(te$pair1, te$pair2))[,1]
```

```
## 313/313 - 0s - 1ms/step
```

``` r
te_acc  <- compute_accuracy(te_pred, te$y)

sprintf('* Accuracy on training set: %0.2f%%', (100 * tr_acc))
```

```
## [1] "* Accuracy on training set: 99.60%"
```

``` r
sprintf('* Accuracy on test set: %0.2f%%', (100 * te_acc))
```

```
## [1] "* Accuracy on test set: 97.81%"
```

## Plots



