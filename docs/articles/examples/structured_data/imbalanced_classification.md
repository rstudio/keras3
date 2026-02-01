# Imbalanced classification: credit card fraud detection

``` r
library(keras3)
use_backend("jax")
```

## Introduction

This example looks at the [Kaggle Credit Card Fraud
Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud/) dataset to
demonstrate how to train a classification model on data with highly
imbalanced classes. You can download the data by clicking “Download” at
the link, or if you’re setup with a kaggle API key at
`"~/.kaggle/kagle.json"`, you can run the following:

``` r
reticulate::py_install("kaggle", pip = TRUE)
reticulate::py_available(TRUE) # ensure 'kaggle' is on the PATH
system("kaggle datasets download -d mlg-ulb/creditcardfraud")
zip::unzip("creditcardfraud.zip", files = "creditcard.csv")
```

## First, load the data

``` r
library(readr)
df <- read_csv("creditcard.csv", col_types = cols(
  Class = col_integer(),
  .default = col_double()
))
tibble::glimpse(df)
```

    ## Rows: 284,807
    ## Columns: 31
    ## $ Time   <dbl> 0, 0, 1, 1, 2, 2, 4, 7, 7, 9, 10, 10, 10, 11, 12, 12, 12, 1…
    ## $ V1     <dbl> -1.3598071, 1.1918571, -1.3583541, -0.9662717, -1.1582331, …
    ## $ V2     <dbl> -0.07278117, 0.26615071, -1.34016307, -0.18522601, 0.877736…
    ## $ V3     <dbl> 2.53634674, 0.16648011, 1.77320934, 1.79299334, 1.54871785,…
    ## $ V4     <dbl> 1.37815522, 0.44815408, 0.37977959, -0.86329128, 0.40303393…
    ## $ V5     <dbl> -0.33832077, 0.06001765, -0.50319813, -0.01030888, -0.40719…
    ## $ V6     <dbl> 0.46238778, -0.08236081, 1.80049938, 1.24720317, 0.09592146…
    ## $ V7     <dbl> 0.239598554, -0.078802983, 0.791460956, 0.237608940, 0.5929…
    ## $ V8     <dbl> 0.098697901, 0.085101655, 0.247675787, 0.377435875, -0.2705…
    ## $ V9     <dbl> 0.3637870, -0.2554251, -1.5146543, -1.3870241, 0.8177393, -…
    ## $ V10    <dbl> 0.09079417, -0.16697441, 0.20764287, -0.05495192, 0.7530744…
    ## $ V11    <dbl> -0.55159953, 1.61272666, 0.62450146, -0.22648726, -0.822842…
    ## $ V12    <dbl> -0.61780086, 1.06523531, 0.06608369, 0.17822823, 0.53819555…
    ## $ V13    <dbl> -0.99138985, 0.48909502, 0.71729273, 0.50775687, 1.34585159…
    ## $ V14    <dbl> -0.31116935, -0.14377230, -0.16594592, -0.28792375, -1.1196…
    ## $ V15    <dbl> 1.468176972, 0.635558093, 2.345864949, -0.631418118, 0.1751…
    ## $ V16    <dbl> -0.47040053, 0.46391704, -2.89008319, -1.05964725, -0.45144…
    ## $ V17    <dbl> 0.207971242, -0.114804663, 1.109969379, -0.684092786, -0.23…
    ## $ V18    <dbl> 0.02579058, -0.18336127, -0.12135931, 1.96577500, -0.038194…
    ## $ V19    <dbl> 0.40399296, -0.14578304, -2.26185710, -1.23262197, 0.803486…
    ## $ V20    <dbl> 0.25141210, -0.06908314, 0.52497973, -0.20803778, 0.4085423…
    ## $ V21    <dbl> -0.018306778, -0.225775248, 0.247998153, -0.108300452, -0.0…
    ## $ V22    <dbl> 0.277837576, -0.638671953, 0.771679402, 0.005273597, 0.7982…
    ## $ V23    <dbl> -0.110473910, 0.101288021, 0.909412262, -0.190320519, -0.13…
    ## $ V24    <dbl> 0.06692807, -0.33984648, -0.68928096, -1.17557533, 0.141266…
    ## $ V25    <dbl> 0.12853936, 0.16717040, -0.32764183, 0.64737603, -0.2060095…
    ## $ V26    <dbl> -0.18911484, 0.12589453, -0.13909657, -0.22192884, 0.502292…
    ## $ V27    <dbl> 0.133558377, -0.008983099, -0.055352794, 0.062722849, 0.219…
    ## $ V28    <dbl> -0.021053053, 0.014724169, -0.059751841, 0.061457629, 0.215…
    ## $ Amount <dbl> 149.62, 2.69, 378.66, 123.50, 69.99, 3.67, 4.99, 40.80, 93.…
    ## $ Class  <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,…

## Prepare a validation set

``` r
val_idx <- nrow(df) %>% sample.int(., round( . * 0.2))
val_df <- df[val_idx, ]
train_df <- df[-val_idx, ]

cat("Number of training samples:", nrow(train_df), "\n")
```

    ## Number of training samples: 227846

``` r
cat("Number of validation samples:", nrow(val_df), "\n")
```

    ## Number of validation samples: 56961

## Analyze class imbalance in the targets

``` r
counts <- table(train_df$Class)
counts
```

    ##
    ##      0      1
    ## 227441    405

``` r
cat(sprintf("Number of positive samples in training data: %i (%.2f%% of total)",
            counts["1"], 100 * counts["1"] / sum(counts)))
```

    ## Number of positive samples in training data: 405 (0.18% of total)

``` r
weight_for_0 = 1 / counts["0"]
weight_for_1 = 1 / counts["1"]
```

## Normalize the data using training set statistics

``` r
feature_names <- colnames(train_df) %>% setdiff("Class")

train_features <- as.matrix(train_df[feature_names])
train_targets <- as.matrix(train_df$Class)

val_features <- as.matrix(val_df[feature_names])
val_targets <- as.matrix(val_df$Class)

train_features %<>% scale()
val_features %<>% scale(center = attr(train_features, "scaled:center"),
                        scale = attr(train_features, "scaled:scale"))
```

## Build a binary classification model

``` r
model <-
  keras_model_sequential(input_shape = ncol(train_features)) |>
  layer_dense(256, activation = "relu") |>
  layer_dense(256, activation = "relu") |>
  layer_dropout(0.3) |>
  layer_dense(256, activation = "relu") |>
  layer_dropout(0.3) |>
  layer_dense(1, activation = "sigmoid")

model
```

    ## Model: "sequential"
    ## ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ## ┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
    ## ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    ## │ dense (Dense)                   │ (None, 256)            │         7,936 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ dense_1 (Dense)                 │ (None, 256)            │        65,792 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ dropout (Dropout)               │ (None, 256)            │             0 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ dense_2 (Dense)                 │ (None, 256)            │        65,792 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ dropout_1 (Dropout)             │ (None, 256)            │             0 │
    ## ├─────────────────────────────────┼────────────────────────┼───────────────┤
    ## │ dense_3 (Dense)                 │ (None, 1)              │           257 │
    ## └─────────────────────────────────┴────────────────────────┴───────────────┘
    ##  Total params: 139,777 (546.00 KB)
    ##  Trainable params: 139,777 (546.00 KB)
    ##  Non-trainable params: 0 (0.00 B)

## Train the model with `class_weight` argument

``` r
metrics <- list(
  metric_false_negatives(name = "fn"),
  metric_false_positives(name = "fp"),
  metric_true_negatives(name = "tn"),
  metric_true_positives(name = "tp"),
  metric_precision(name = "precision"),
  metric_recall(name = "recall")
)
model |> compile(
  optimizer = optimizer_adam(1e-2),
  loss = "binary_crossentropy",
  metrics = metrics
)
callbacks <- list(
  callback_model_checkpoint("fraud_model_at_epoch_{epoch}.keras")
)

class_weight <- list("0" = weight_for_0,
                     "1" = weight_for_1)

model |> fit(
  train_features, train_targets,
  validation_data = list(val_features, val_targets),
  class_weight = class_weight,
  batch_size = 2048,
  epochs = 30,
  callbacks = callbacks,
  verbose = 2
)
```

    ## Epoch 1/30
    ## 112/112 - 5s - 45ms/step - fn: 45.0000 - fp: 30771.0000 - loss: 2.3358e-06 - precision: 0.0116 - recall: 0.8889 - tn: 196670.0000 - tp: 360.0000 - val_fn: 12.0000 - val_fp: 608.0000 - val_loss: 0.0627 - val_precision: 0.1098 - val_recall: 0.8621 - val_tn: 56266.0000 - val_tp: 75.0000
    ## Epoch 2/30
    ## 112/112 - 0s - 2ms/step - fn: 36.0000 - fp: 8897.0000 - loss: 1.4359e-06 - precision: 0.0398 - recall: 0.9111 - tn: 218544.0000 - tp: 369.0000 - val_fn: 10.0000 - val_fp: 991.0000 - val_loss: 0.0873 - val_precision: 0.0721 - val_recall: 0.8851 - val_tn: 55883.0000 - val_tp: 77.0000
    ## Epoch 3/30
    ## 112/112 - 0s - 2ms/step - fn: 31.0000 - fp: 6908.0000 - loss: 1.0973e-06 - precision: 0.0514 - recall: 0.9235 - tn: 220533.0000 - tp: 374.0000 - val_fn: 6.0000 - val_fp: 2546.0000 - val_loss: 0.1261 - val_precision: 0.0308 - val_recall: 0.9310 - val_tn: 54328.0000 - val_tp: 81.0000
    ## Epoch 4/30
    ## 112/112 - 0s - 2ms/step - fn: 23.0000 - fp: 8297.0000 - loss: 1.0339e-06 - precision: 0.0440 - recall: 0.9432 - tn: 219144.0000 - tp: 382.0000 - val_fn: 10.0000 - val_fp: 1205.0000 - val_loss: 0.0656 - val_precision: 0.0601 - val_recall: 0.8851 - val_tn: 55669.0000 - val_tp: 77.0000
    ## Epoch 5/30
    ## 112/112 - 0s - 2ms/step - fn: 17.0000 - fp: 6667.0000 - loss: 7.6904e-07 - precision: 0.0550 - recall: 0.9580 - tn: 220774.0000 - tp: 388.0000 - val_fn: 8.0000 - val_fp: 910.0000 - val_loss: 0.0443 - val_precision: 0.0799 - val_recall: 0.9080 - val_tn: 55964.0000 - val_tp: 79.0000
    ## Epoch 6/30
    ## 112/112 - 0s - 2ms/step - fn: 19.0000 - fp: 8230.0000 - loss: 8.4087e-07 - precision: 0.0448 - recall: 0.9531 - tn: 219211.0000 - tp: 386.0000 - val_fn: 9.0000 - val_fp: 1516.0000 - val_loss: 0.0682 - val_precision: 0.0489 - val_recall: 0.8966 - val_tn: 55358.0000 - val_tp: 78.0000
    ## Epoch 7/30
    ## 112/112 - 0s - 2ms/step - fn: 13.0000 - fp: 6909.0000 - loss: 6.5042e-07 - precision: 0.0537 - recall: 0.9679 - tn: 220532.0000 - tp: 392.0000 - val_fn: 9.0000 - val_fp: 586.0000 - val_loss: 0.0338 - val_precision: 0.1175 - val_recall: 0.8966 - val_tn: 56288.0000 - val_tp: 78.0000
    ## Epoch 8/30
    ## 112/112 - 0s - 2ms/step - fn: 16.0000 - fp: 7808.0000 - loss: 7.0694e-07 - precision: 0.0475 - recall: 0.9605 - tn: 219633.0000 - tp: 389.0000 - val_fn: 10.0000 - val_fp: 1165.0000 - val_loss: 0.0517 - val_precision: 0.0620 - val_recall: 0.8851 - val_tn: 55709.0000 - val_tp: 77.0000
    ## Epoch 9/30
    ## 112/112 - 0s - 2ms/step - fn: 14.0000 - fp: 6071.0000 - loss: 6.8702e-07 - precision: 0.0605 - recall: 0.9654 - tn: 221370.0000 - tp: 391.0000 - val_fn: 7.0000 - val_fp: 1749.0000 - val_loss: 0.0648 - val_precision: 0.0437 - val_recall: 0.9195 - val_tn: 55125.0000 - val_tp: 80.0000
    ## Epoch 10/30
    ## 112/112 - 0s - 2ms/step - fn: 10.0000 - fp: 7065.0000 - loss: 6.1083e-07 - precision: 0.0529 - recall: 0.9753 - tn: 220376.0000 - tp: 395.0000 - val_fn: 6.0000 - val_fp: 3934.0000 - val_loss: 0.1280 - val_precision: 0.0202 - val_recall: 0.9310 - val_tn: 52940.0000 - val_tp: 81.0000
    ## Epoch 11/30
    ## 112/112 - 0s - 2ms/step - fn: 8.0000 - fp: 7531.0000 - loss: 5.9298e-07 - precision: 0.0501 - recall: 0.9802 - tn: 219910.0000 - tp: 397.0000 - val_fn: 8.0000 - val_fp: 2522.0000 - val_loss: 0.1036 - val_precision: 0.0304 - val_recall: 0.9080 - val_tn: 54352.0000 - val_tp: 79.0000
    ## Epoch 12/30
    ## 112/112 - 0s - 2ms/step - fn: 10.0000 - fp: 7268.0000 - loss: 5.6487e-07 - precision: 0.0515 - recall: 0.9753 - tn: 220173.0000 - tp: 395.0000 - val_fn: 9.0000 - val_fp: 2106.0000 - val_loss: 0.0742 - val_precision: 0.0357 - val_recall: 0.8966 - val_tn: 54768.0000 - val_tp: 78.0000
    ## Epoch 13/30
    ## 112/112 - 0s - 2ms/step - fn: 8.0000 - fp: 6029.0000 - loss: 4.8386e-07 - precision: 0.0618 - recall: 0.9802 - tn: 221412.0000 - tp: 397.0000 - val_fn: 9.0000 - val_fp: 2326.0000 - val_loss: 0.0841 - val_precision: 0.0324 - val_recall: 0.8966 - val_tn: 54548.0000 - val_tp: 78.0000
    ## Epoch 14/30
    ## 112/112 - 0s - 2ms/step - fn: 13.0000 - fp: 9693.0000 - loss: 1.0115e-06 - precision: 0.0389 - recall: 0.9679 - tn: 217748.0000 - tp: 392.0000 - val_fn: 4.0000 - val_fp: 5328.0000 - val_loss: 0.2307 - val_precision: 0.0153 - val_recall: 0.9540 - val_tn: 51546.0000 - val_tp: 83.0000
    ## Epoch 15/30
    ## 112/112 - 0s - 2ms/step - fn: 11.0000 - fp: 6654.0000 - loss: 7.9630e-07 - precision: 0.0559 - recall: 0.9728 - tn: 220787.0000 - tp: 394.0000 - val_fn: 8.0000 - val_fp: 2120.0000 - val_loss: 0.3231 - val_precision: 0.0359 - val_recall: 0.9080 - val_tn: 54754.0000 - val_tp: 79.0000
    ## Epoch 16/30
    ## 112/112 - 0s - 2ms/step - fn: 17.0000 - fp: 8922.0000 - loss: 1.3151e-06 - precision: 0.0417 - recall: 0.9580 - tn: 218519.0000 - tp: 388.0000 - val_fn: 9.0000 - val_fp: 1983.0000 - val_loss: 0.0852 - val_precision: 0.0378 - val_recall: 0.8966 - val_tn: 54891.0000 - val_tp: 78.0000
    ## Epoch 17/30
    ## 112/112 - 0s - 2ms/step - fn: 8.0000 - fp: 4879.0000 - loss: 5.1664e-07 - precision: 0.0752 - recall: 0.9802 - tn: 222562.0000 - tp: 397.0000 - val_fn: 6.0000 - val_fp: 1399.0000 - val_loss: 0.0657 - val_precision: 0.0547 - val_recall: 0.9310 - val_tn: 55475.0000 - val_tp: 81.0000
    ## Epoch 18/30
    ## 112/112 - 0s - 2ms/step - fn: 6.0000 - fp: 4272.0000 - loss: 3.8395e-07 - precision: 0.0854 - recall: 0.9852 - tn: 223169.0000 - tp: 399.0000 - val_fn: 11.0000 - val_fp: 905.0000 - val_loss: 0.0403 - val_precision: 0.0775 - val_recall: 0.8736 - val_tn: 55969.0000 - val_tp: 76.0000
    ## Epoch 19/30
    ## 112/112 - 0s - 2ms/step - fn: 5.0000 - fp: 4217.0000 - loss: 4.2943e-07 - precision: 0.0866 - recall: 0.9877 - tn: 223224.0000 - tp: 400.0000 - val_fn: 10.0000 - val_fp: 1006.0000 - val_loss: 0.0422 - val_precision: 0.0711 - val_recall: 0.8851 - val_tn: 55868.0000 - val_tp: 77.0000
    ## Epoch 20/30
    ## 112/112 - 0s - 2ms/step - fn: 1.0000 - fp: 3221.0000 - loss: 2.6898e-07 - precision: 0.1114 - recall: 0.9975 - tn: 224220.0000 - tp: 404.0000 - val_fn: 11.0000 - val_fp: 385.0000 - val_loss: 0.0207 - val_precision: 0.1649 - val_recall: 0.8736 - val_tn: 56489.0000 - val_tp: 76.0000
    ## Epoch 21/30
    ## 112/112 - 0s - 2ms/step - fn: 1.0000 - fp: 2307.0000 - loss: 2.2705e-07 - precision: 0.1490 - recall: 0.9975 - tn: 225134.0000 - tp: 404.0000 - val_fn: 11.0000 - val_fp: 827.0000 - val_loss: 0.0487 - val_precision: 0.0842 - val_recall: 0.8736 - val_tn: 56047.0000 - val_tp: 76.0000
    ## Epoch 22/30
    ## 112/112 - 0s - 2ms/step - fn: 5.0000 - fp: 4143.0000 - loss: 4.2077e-07 - precision: 0.0880 - recall: 0.9877 - tn: 223298.0000 - tp: 400.0000 - val_fn: 8.0000 - val_fp: 1154.0000 - val_loss: 0.0378 - val_precision: 0.0641 - val_recall: 0.9080 - val_tn: 55720.0000 - val_tp: 79.0000
    ## Epoch 23/30
    ## 112/112 - 0s - 2ms/step - fn: 4.0000 - fp: 5186.0000 - loss: 4.2932e-07 - precision: 0.0718 - recall: 0.9901 - tn: 222255.0000 - tp: 401.0000 - val_fn: 10.0000 - val_fp: 1483.0000 - val_loss: 0.0677 - val_precision: 0.0494 - val_recall: 0.8851 - val_tn: 55391.0000 - val_tp: 77.0000
    ## Epoch 24/30
    ## 112/112 - 0s - 2ms/step - fn: 3.0000 - fp: 3535.0000 - loss: 2.9635e-07 - precision: 0.1021 - recall: 0.9926 - tn: 223906.0000 - tp: 402.0000 - val_fn: 9.0000 - val_fp: 1102.0000 - val_loss: 0.0479 - val_precision: 0.0661 - val_recall: 0.8966 - val_tn: 55772.0000 - val_tp: 78.0000
    ## Epoch 25/30
    ## 112/112 - 0s - 2ms/step - fn: 1.0000 - fp: 3044.0000 - loss: 2.7673e-07 - precision: 0.1172 - recall: 0.9975 - tn: 224397.0000 - tp: 404.0000 - val_fn: 10.0000 - val_fp: 1688.0000 - val_loss: 0.1471 - val_precision: 0.0436 - val_recall: 0.8851 - val_tn: 55186.0000 - val_tp: 77.0000
    ## Epoch 26/30
    ## 112/112 - 0s - 2ms/step - fn: 4.0000 - fp: 4932.0000 - loss: 5.1021e-07 - precision: 0.0752 - recall: 0.9901 - tn: 222509.0000 - tp: 401.0000 - val_fn: 9.0000 - val_fp: 1723.0000 - val_loss: 0.0602 - val_precision: 0.0433 - val_recall: 0.8966 - val_tn: 55151.0000 - val_tp: 78.0000
    ## Epoch 27/30
    ## 112/112 - 0s - 2ms/step - fn: 8.0000 - fp: 7264.0000 - loss: 8.2890e-07 - precision: 0.0518 - recall: 0.9802 - tn: 220177.0000 - tp: 397.0000 - val_fn: 9.0000 - val_fp: 1075.0000 - val_loss: 0.0687 - val_precision: 0.0676 - val_recall: 0.8966 - val_tn: 55799.0000 - val_tp: 78.0000
    ## Epoch 28/30
    ## 112/112 - 0s - 2ms/step - fn: 3.0000 - fp: 3530.0000 - loss: 3.5171e-07 - precision: 0.1022 - recall: 0.9926 - tn: 223911.0000 - tp: 402.0000 - val_fn: 10.0000 - val_fp: 813.0000 - val_loss: 0.0485 - val_precision: 0.0865 - val_recall: 0.8851 - val_tn: 56061.0000 - val_tp: 77.0000
    ## Epoch 29/30
    ## 112/112 - 0s - 2ms/step - fn: 1.0000 - fp: 1671.0000 - loss: 1.7981e-07 - precision: 0.1947 - recall: 0.9975 - tn: 225770.0000 - tp: 404.0000 - val_fn: 11.0000 - val_fp: 354.0000 - val_loss: 0.0264 - val_precision: 0.1767 - val_recall: 0.8736 - val_tn: 56520.0000 - val_tp: 76.0000
    ## Epoch 30/30
    ## 112/112 - 0s - 2ms/step - fn: 2.0000 - fp: 1495.0000 - loss: 1.7800e-07 - precision: 0.2123 - recall: 0.9951 - tn: 225946.0000 - tp: 403.0000 - val_fn: 8.0000 - val_fp: 1980.0000 - val_loss: 0.1391 - val_precision: 0.0384 - val_recall: 0.9080 - val_tn: 54894.0000 - val_tp: 79.0000

``` r
val_pred <- model %>%
  predict(val_features) %>%
  { as.integer(. > 0.5) }
```

    ## 1781/1781 - 1s - 584us/step

``` r
pred_correct <- val_df$Class == val_pred
cat(sprintf("Validation accuracy: %.2f", mean(pred_correct)))
```

    ## Validation accuracy: 0.97

``` r
fraudulent <- val_df$Class == 1

n_fraudulent_detected <- sum(fraudulent & pred_correct)
n_fraudulent_missed <- sum(fraudulent & !pred_correct)
n_legitimate_flagged <- sum(!fraudulent & !pred_correct)
```

## Conclusions

At the end of training, out of 56,961 validation transactions, we are:

- Correctly identifying 79 of them as fraudulent
- Missing 8 fraudulent transactions
- At the cost of incorrectly flagging 1,980 legitimate transactions

In the real world, one would put an even higher weight on class 1, so as
to reflect that False Negatives are more costly than False Positives.

Next time your credit card gets declined in an online purchase – this is
why.
