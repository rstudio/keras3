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
    ## 227454    392

``` r
cat(sprintf("Number of positive samples in training data: %i (%.2f%% of total)",
            counts["1"], 100 * counts["1"] / sum(counts)))
```

    ## Number of positive samples in training data: 392 (0.17% of total)

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
    ## 112/112 - 5s - 44ms/step - fn: 42.0000 - fp: 19532.0000 - loss: 2.1752e-06 - precision: 0.0176 - recall: 0.8929 - tn: 207922.0000 - tp: 350.0000 - val_fn: 9.0000 - val_fp: 1413.0000 - val_loss: 0.1081 - val_precision: 0.0605 - val_recall: 0.9100 - val_tn: 55448.0000 - val_tp: 91.0000
    ## Epoch 2/30
    ## 112/112 - 0s - 2ms/step - fn: 31.0000 - fp: 8753.0000 - loss: 1.3321e-06 - precision: 0.0396 - recall: 0.9209 - tn: 218701.0000 - tp: 361.0000 - val_fn: 7.0000 - val_fp: 1092.0000 - val_loss: 0.0625 - val_precision: 0.0785 - val_recall: 0.9300 - val_tn: 55769.0000 - val_tp: 93.0000
    ## Epoch 3/30
    ## 112/112 - 0s - 2ms/step - fn: 27.0000 - fp: 7906.0000 - loss: 1.1587e-06 - precision: 0.0441 - recall: 0.9311 - tn: 219548.0000 - tp: 365.0000 - val_fn: 5.0000 - val_fp: 2520.0000 - val_loss: 0.1403 - val_precision: 0.0363 - val_recall: 0.9500 - val_tn: 54341.0000 - val_tp: 95.0000
    ## Epoch 4/30
    ## 112/112 - 0s - 2ms/step - fn: 26.0000 - fp: 8828.0000 - loss: 9.6770e-07 - precision: 0.0398 - recall: 0.9337 - tn: 218626.0000 - tp: 366.0000 - val_fn: 7.0000 - val_fp: 1248.0000 - val_loss: 0.0534 - val_precision: 0.0694 - val_recall: 0.9300 - val_tn: 55613.0000 - val_tp: 93.0000
    ## Epoch 5/30
    ## 112/112 - 0s - 2ms/step - fn: 14.0000 - fp: 7833.0000 - loss: 7.9722e-07 - precision: 0.0460 - recall: 0.9643 - tn: 219621.0000 - tp: 378.0000 - val_fn: 7.0000 - val_fp: 1194.0000 - val_loss: 0.0626 - val_precision: 0.0723 - val_recall: 0.9300 - val_tn: 55667.0000 - val_tp: 93.0000
    ## Epoch 6/30
    ## 112/112 - 0s - 2ms/step - fn: 14.0000 - fp: 7123.0000 - loss: 6.8667e-07 - precision: 0.0504 - recall: 0.9643 - tn: 220331.0000 - tp: 378.0000 - val_fn: 8.0000 - val_fp: 2032.0000 - val_loss: 0.1144 - val_precision: 0.0433 - val_recall: 0.9200 - val_tn: 54829.0000 - val_tp: 92.0000
    ## Epoch 7/30
    ## 112/112 - 0s - 2ms/step - fn: 17.0000 - fp: 10371.0000 - loss: 8.7018e-07 - precision: 0.0349 - recall: 0.9566 - tn: 217083.0000 - tp: 375.0000 - val_fn: 7.0000 - val_fp: 2574.0000 - val_loss: 0.0890 - val_precision: 0.0349 - val_recall: 0.9300 - val_tn: 54287.0000 - val_tp: 93.0000
    ## Epoch 8/30
    ## 112/112 - 0s - 2ms/step - fn: 21.0000 - fp: 9461.0000 - loss: 9.6619e-07 - precision: 0.0377 - recall: 0.9464 - tn: 217993.0000 - tp: 371.0000 - val_fn: 8.0000 - val_fp: 2140.0000 - val_loss: 0.0830 - val_precision: 0.0412 - val_recall: 0.9200 - val_tn: 54721.0000 - val_tp: 92.0000
    ## Epoch 9/30
    ## 112/112 - 0s - 2ms/step - fn: 16.0000 - fp: 8049.0000 - loss: 7.2046e-07 - precision: 0.0446 - recall: 0.9592 - tn: 219405.0000 - tp: 376.0000 - val_fn: 6.0000 - val_fp: 2400.0000 - val_loss: 0.1114 - val_precision: 0.0377 - val_recall: 0.9400 - val_tn: 54461.0000 - val_tp: 94.0000
    ## Epoch 10/30
    ## 112/112 - 0s - 2ms/step - fn: 9.0000 - fp: 6700.0000 - loss: 5.7529e-07 - precision: 0.0541 - recall: 0.9770 - tn: 220754.0000 - tp: 383.0000 - val_fn: 9.0000 - val_fp: 1134.0000 - val_loss: 0.0455 - val_precision: 0.0743 - val_recall: 0.9100 - val_tn: 55727.0000 - val_tp: 91.0000
    ## Epoch 11/30
    ## 112/112 - 0s - 2ms/step - fn: 8.0000 - fp: 5922.0000 - loss: 5.6104e-07 - precision: 0.0609 - recall: 0.9796 - tn: 221532.0000 - tp: 384.0000 - val_fn: 6.0000 - val_fp: 2725.0000 - val_loss: 0.0940 - val_precision: 0.0333 - val_recall: 0.9400 - val_tn: 54136.0000 - val_tp: 94.0000
    ## Epoch 12/30
    ## 112/112 - 0s - 2ms/step - fn: 6.0000 - fp: 6668.0000 - loss: 4.4606e-07 - precision: 0.0547 - recall: 0.9847 - tn: 220786.0000 - tp: 386.0000 - val_fn: 6.0000 - val_fp: 1938.0000 - val_loss: 0.0835 - val_precision: 0.0463 - val_recall: 0.9400 - val_tn: 54923.0000 - val_tp: 94.0000
    ## Epoch 13/30
    ## 112/112 - 0s - 2ms/step - fn: 4.0000 - fp: 5156.0000 - loss: 4.3178e-07 - precision: 0.0700 - recall: 0.9898 - tn: 222298.0000 - tp: 388.0000 - val_fn: 6.0000 - val_fp: 2764.0000 - val_loss: 0.1406 - val_precision: 0.0329 - val_recall: 0.9400 - val_tn: 54097.0000 - val_tp: 94.0000
    ## Epoch 14/30
    ## 112/112 - 0s - 2ms/step - fn: 5.0000 - fp: 6519.0000 - loss: 5.1981e-07 - precision: 0.0560 - recall: 0.9872 - tn: 220935.0000 - tp: 387.0000 - val_fn: 7.0000 - val_fp: 2026.0000 - val_loss: 0.0811 - val_precision: 0.0439 - val_recall: 0.9300 - val_tn: 54835.0000 - val_tp: 93.0000
    ## Epoch 15/30
    ## 112/112 - 0s - 2ms/step - fn: 9.0000 - fp: 6366.0000 - loss: 4.7680e-07 - precision: 0.0567 - recall: 0.9770 - tn: 221088.0000 - tp: 383.0000 - val_fn: 8.0000 - val_fp: 922.0000 - val_loss: 0.0387 - val_precision: 0.0907 - val_recall: 0.9200 - val_tn: 55939.0000 - val_tp: 92.0000
    ## Epoch 16/30
    ## 112/112 - 0s - 2ms/step - fn: 4.0000 - fp: 4320.0000 - loss: 3.2840e-07 - precision: 0.0824 - recall: 0.9898 - tn: 223134.0000 - tp: 388.0000 - val_fn: 8.0000 - val_fp: 700.0000 - val_loss: 0.0357 - val_precision: 0.1162 - val_recall: 0.9200 - val_tn: 56161.0000 - val_tp: 92.0000
    ## Epoch 17/30
    ## 112/112 - 0s - 2ms/step - fn: 3.0000 - fp: 3367.0000 - loss: 3.0205e-07 - precision: 0.1036 - recall: 0.9923 - tn: 224087.0000 - tp: 389.0000 - val_fn: 6.0000 - val_fp: 2152.0000 - val_loss: 0.0967 - val_precision: 0.0419 - val_recall: 0.9400 - val_tn: 54709.0000 - val_tp: 94.0000
    ## Epoch 18/30
    ## 112/112 - 0s - 2ms/step - fn: 4.0000 - fp: 4130.0000 - loss: 3.0376e-07 - precision: 0.0859 - recall: 0.9898 - tn: 223324.0000 - tp: 388.0000 - val_fn: 9.0000 - val_fp: 1005.0000 - val_loss: 0.0461 - val_precision: 0.0830 - val_recall: 0.9100 - val_tn: 55856.0000 - val_tp: 91.0000
    ## Epoch 19/30
    ## 112/112 - 0s - 2ms/step - fn: 8.0000 - fp: 9140.0000 - loss: 7.5488e-07 - precision: 0.0403 - recall: 0.9796 - tn: 218314.0000 - tp: 384.0000 - val_fn: 9.0000 - val_fp: 2019.0000 - val_loss: 0.0787 - val_precision: 0.0431 - val_recall: 0.9100 - val_tn: 54842.0000 - val_tp: 91.0000
    ## Epoch 20/30
    ## 112/112 - 0s - 2ms/step - fn: 2.0000 - fp: 4394.0000 - loss: 3.4110e-07 - precision: 0.0815 - recall: 0.9949 - tn: 223060.0000 - tp: 390.0000 - val_fn: 7.0000 - val_fp: 949.0000 - val_loss: 0.0405 - val_precision: 0.0893 - val_recall: 0.9300 - val_tn: 55912.0000 - val_tp: 93.0000
    ## Epoch 21/30
    ## 112/112 - 0s - 2ms/step - fn: 5.0000 - fp: 5075.0000 - loss: 4.7739e-07 - precision: 0.0709 - recall: 0.9872 - tn: 222379.0000 - tp: 387.0000 - val_fn: 6.0000 - val_fp: 4845.0000 - val_loss: 0.1750 - val_precision: 0.0190 - val_recall: 0.9400 - val_tn: 52016.0000 - val_tp: 94.0000
    ## Epoch 22/30
    ## 112/112 - 0s - 2ms/step - fn: 3.0000 - fp: 6371.0000 - loss: 4.1728e-07 - precision: 0.0575 - recall: 0.9923 - tn: 221083.0000 - tp: 389.0000 - val_fn: 9.0000 - val_fp: 734.0000 - val_loss: 0.0358 - val_precision: 0.1103 - val_recall: 0.9100 - val_tn: 56127.0000 - val_tp: 91.0000
    ## Epoch 23/30
    ## 112/112 - 0s - 2ms/step - fn: 11.0000 - fp: 7624.0000 - loss: 8.0021e-07 - precision: 0.0476 - recall: 0.9719 - tn: 219830.0000 - tp: 381.0000 - val_fn: 4.0000 - val_fp: 5131.0000 - val_loss: 0.2242 - val_precision: 0.0184 - val_recall: 0.9600 - val_tn: 51730.0000 - val_tp: 96.0000
    ## Epoch 24/30
    ## 112/112 - 0s - 2ms/step - fn: 8.0000 - fp: 5914.0000 - loss: 5.3629e-07 - precision: 0.0610 - recall: 0.9796 - tn: 221540.0000 - tp: 384.0000 - val_fn: 8.0000 - val_fp: 1757.0000 - val_loss: 0.0760 - val_precision: 0.0498 - val_recall: 0.9200 - val_tn: 55104.0000 - val_tp: 92.0000
    ## Epoch 25/30
    ## 112/112 - 0s - 2ms/step - fn: 7.0000 - fp: 5386.0000 - loss: 4.1623e-07 - precision: 0.0667 - recall: 0.9821 - tn: 222068.0000 - tp: 385.0000 - val_fn: 9.0000 - val_fp: 823.0000 - val_loss: 0.0384 - val_precision: 0.0996 - val_recall: 0.9100 - val_tn: 56038.0000 - val_tp: 91.0000
    ## Epoch 26/30
    ## 112/112 - 0s - 2ms/step - fn: 3.0000 - fp: 3474.0000 - loss: 2.9035e-07 - precision: 0.1007 - recall: 0.9923 - tn: 223980.0000 - tp: 389.0000 - val_fn: 9.0000 - val_fp: 1152.0000 - val_loss: 0.0631 - val_precision: 0.0732 - val_recall: 0.9100 - val_tn: 55709.0000 - val_tp: 91.0000
    ## Epoch 27/30
    ## 112/112 - 0s - 2ms/step - fn: 5.0000 - fp: 4425.0000 - loss: 3.9144e-07 - precision: 0.0804 - recall: 0.9872 - tn: 223029.0000 - tp: 387.0000 - val_fn: 8.0000 - val_fp: 1424.0000 - val_loss: 0.0641 - val_precision: 0.0607 - val_recall: 0.9200 - val_tn: 55437.0000 - val_tp: 92.0000
    ## Epoch 28/30
    ## 112/112 - 0s - 2ms/step - fn: 4.0000 - fp: 3440.0000 - loss: 2.7712e-07 - precision: 0.1014 - recall: 0.9898 - tn: 224014.0000 - tp: 388.0000 - val_fn: 9.0000 - val_fp: 485.0000 - val_loss: 0.0260 - val_precision: 0.1580 - val_recall: 0.9100 - val_tn: 56376.0000 - val_tp: 91.0000
    ## Epoch 29/30
    ## 112/112 - 0s - 2ms/step - fn: 2.0000 - fp: 2694.0000 - loss: 2.2238e-07 - precision: 0.1265 - recall: 0.9949 - tn: 224760.0000 - tp: 390.0000 - val_fn: 10.0000 - val_fp: 671.0000 - val_loss: 0.0332 - val_precision: 0.1183 - val_recall: 0.9000 - val_tn: 56190.0000 - val_tp: 90.0000
    ## Epoch 30/30
    ## 112/112 - 0s - 2ms/step - fn: 3.0000 - fp: 3213.0000 - loss: 2.7529e-07 - precision: 0.1080 - recall: 0.9923 - tn: 224241.0000 - tp: 389.0000 - val_fn: 9.0000 - val_fp: 743.0000 - val_loss: 0.0344 - val_precision: 0.1091 - val_recall: 0.9100 - val_tn: 56118.0000 - val_tp: 91.0000

``` r
val_pred <- model %>%
  predict(val_features) %>%
  { as.integer(. > 0.5) }
```

    ## 1781/1781 - 1s - 596us/step

``` r
pred_correct <- val_df$Class == val_pred
cat(sprintf("Validation accuracy: %.2f", mean(pred_correct)))
```

    ## Validation accuracy: 0.99

``` r
fraudulent <- val_df$Class == 1

n_fraudulent_detected <- sum(fraudulent & pred_correct)
n_fraudulent_missed <- sum(fraudulent & !pred_correct)
n_legitimate_flagged <- sum(!fraudulent & !pred_correct)
```

## Conclusions

At the end of training, out of 56,961 validation transactions, we are:

- Correctly identifying 91 of them as fraudulent
- Missing 9 fraudulent transactions
- At the cost of incorrectly flagging 743 legitimate transactions

In the real world, one would put an even higher weight on class 1, so as
to reflect that False Negatives are more costly than False Positives.

Next time your credit card gets declined in an online purchase – this is
why.
