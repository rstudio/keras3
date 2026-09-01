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
    ## 227450    396

``` r
cat(sprintf("Number of positive samples in training data: %i (%.2f%% of total)",
            counts["1"], 100 * counts["1"] / sum(counts)))
```

    ## Number of positive samples in training data: 396 (0.17% of total)

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
    ## 112/112 - 2s - 21ms/step - fn: 43.0000 - fp: 26935.0000 - loss: 2.1806e-06 - precision: 0.0129 - recall: 0.8914 - tn: 200515.0000 - tp: 353.0000 - val_fn: 9.0000 - val_fp: 1841.0000 - val_loss: 0.1055 - val_precision: 0.0451 - val_recall: 0.9062 - val_tn: 55024.0000 - val_tp: 87.0000
    ## Epoch 2/30
    ## 112/112 - 0s - 3ms/step - fn: 30.0000 - fp: 9416.0000 - loss: 1.3264e-06 - precision: 0.0374 - recall: 0.9242 - tn: 218034.0000 - tp: 366.0000 - val_fn: 11.0000 - val_fp: 326.0000 - val_loss: 0.0423 - val_precision: 0.2068 - val_recall: 0.8854 - val_tn: 56539.0000 - val_tp: 85.0000
    ## Epoch 3/30
    ## 112/112 - 0s - 3ms/step - fn: 27.0000 - fp: 8597.0000 - loss: 1.1686e-06 - precision: 0.0412 - recall: 0.9318 - tn: 218853.0000 - tp: 369.0000 - val_fn: 10.0000 - val_fp: 1918.0000 - val_loss: 0.1077 - val_precision: 0.0429 - val_recall: 0.8958 - val_tn: 54947.0000 - val_tp: 86.0000
    ## Epoch 4/30
    ## 112/112 - 0s - 3ms/step - fn: 24.0000 - fp: 8762.0000 - loss: 1.2178e-06 - precision: 0.0407 - recall: 0.9394 - tn: 218688.0000 - tp: 372.0000 - val_fn: 7.0000 - val_fp: 3727.0000 - val_loss: 0.1778 - val_precision: 0.0233 - val_recall: 0.9271 - val_tn: 53138.0000 - val_tp: 89.0000
    ## Epoch 5/30
    ## 112/112 - 0s - 2ms/step - fn: 22.0000 - fp: 8645.0000 - loss: 9.2373e-07 - precision: 0.0415 - recall: 0.9444 - tn: 218805.0000 - tp: 374.0000 - val_fn: 6.0000 - val_fp: 4696.0000 - val_loss: 0.2202 - val_precision: 0.0188 - val_recall: 0.9375 - val_tn: 52169.0000 - val_tp: 90.0000
    ## Epoch 6/30
    ## 112/112 - 0s - 3ms/step - fn: 15.0000 - fp: 8411.0000 - loss: 9.0694e-07 - precision: 0.0433 - recall: 0.9621 - tn: 219039.0000 - tp: 381.0000 - val_fn: 8.0000 - val_fp: 1213.0000 - val_loss: 0.0630 - val_precision: 0.0676 - val_recall: 0.9167 - val_tn: 55652.0000 - val_tp: 88.0000
    ## Epoch 7/30
    ## 112/112 - 0s - 3ms/step - fn: 15.0000 - fp: 7904.0000 - loss: 8.2602e-07 - precision: 0.0460 - recall: 0.9621 - tn: 219546.0000 - tp: 381.0000 - val_fn: 9.0000 - val_fp: 2745.0000 - val_loss: 0.1131 - val_precision: 0.0307 - val_recall: 0.9062 - val_tn: 54120.0000 - val_tp: 87.0000
    ## Epoch 8/30
    ## 112/112 - 0s - 3ms/step - fn: 15.0000 - fp: 7071.0000 - loss: 6.6437e-07 - precision: 0.0511 - recall: 0.9621 - tn: 220379.0000 - tp: 381.0000 - val_fn: 8.0000 - val_fp: 1570.0000 - val_loss: 0.0679 - val_precision: 0.0531 - val_recall: 0.9167 - val_tn: 55295.0000 - val_tp: 88.0000
    ## Epoch 9/30
    ## 112/112 - 0s - 3ms/step - fn: 9.0000 - fp: 6058.0000 - loss: 5.5925e-07 - precision: 0.0600 - recall: 0.9773 - tn: 221392.0000 - tp: 387.0000 - val_fn: 6.0000 - val_fp: 2640.0000 - val_loss: 0.0985 - val_precision: 0.0330 - val_recall: 0.9375 - val_tn: 54225.0000 - val_tp: 90.0000
    ## Epoch 10/30
    ## 112/112 - 0s - 2ms/step - fn: 14.0000 - fp: 7870.0000 - loss: 7.3961e-07 - precision: 0.0463 - recall: 0.9646 - tn: 219580.0000 - tp: 382.0000 - val_fn: 6.0000 - val_fp: 5015.0000 - val_loss: 0.2330 - val_precision: 0.0176 - val_recall: 0.9375 - val_tn: 51850.0000 - val_tp: 90.0000
    ## Epoch 11/30
    ## 112/112 - 0s - 3ms/step - fn: 15.0000 - fp: 6965.0000 - loss: 7.3133e-07 - precision: 0.0519 - recall: 0.9621 - tn: 220485.0000 - tp: 381.0000 - val_fn: 8.0000 - val_fp: 1620.0000 - val_loss: 0.0771 - val_precision: 0.0515 - val_recall: 0.9167 - val_tn: 55245.0000 - val_tp: 88.0000
    ## Epoch 12/30
    ## 112/112 - 0s - 3ms/step - fn: 17.0000 - fp: 7172.0000 - loss: 8.1422e-07 - precision: 0.0502 - recall: 0.9571 - tn: 220278.0000 - tp: 379.0000 - val_fn: 8.0000 - val_fp: 2139.0000 - val_loss: 0.0737 - val_precision: 0.0395 - val_recall: 0.9167 - val_tn: 54726.0000 - val_tp: 88.0000
    ## Epoch 13/30
    ## 112/112 - 0s - 3ms/step - fn: 11.0000 - fp: 7801.0000 - loss: 8.0239e-07 - precision: 0.0470 - recall: 0.9722 - tn: 219649.0000 - tp: 385.0000 - val_fn: 10.0000 - val_fp: 1123.0000 - val_loss: 0.0550 - val_precision: 0.0711 - val_recall: 0.8958 - val_tn: 55742.0000 - val_tp: 86.0000
    ## Epoch 14/30
    ## 112/112 - 0s - 3ms/step - fn: 8.0000 - fp: 6287.0000 - loss: 5.9569e-07 - precision: 0.0581 - recall: 0.9798 - tn: 221163.0000 - tp: 388.0000 - val_fn: 9.0000 - val_fp: 1562.0000 - val_loss: 0.0578 - val_precision: 0.0528 - val_recall: 0.9062 - val_tn: 55303.0000 - val_tp: 87.0000
    ## Epoch 15/30
    ## 112/112 - 0s - 3ms/step - fn: 4.0000 - fp: 4538.0000 - loss: 3.6784e-07 - precision: 0.0795 - recall: 0.9899 - tn: 222912.0000 - tp: 392.0000 - val_fn: 9.0000 - val_fp: 1334.0000 - val_loss: 0.0523 - val_precision: 0.0612 - val_recall: 0.9062 - val_tn: 55531.0000 - val_tp: 87.0000
    ## Epoch 16/30
    ## 112/112 - 0s - 3ms/step - fn: 6.0000 - fp: 4920.0000 - loss: 4.1569e-07 - precision: 0.0734 - recall: 0.9848 - tn: 222530.0000 - tp: 390.0000 - val_fn: 9.0000 - val_fp: 1698.0000 - val_loss: 0.0723 - val_precision: 0.0487 - val_recall: 0.9062 - val_tn: 55167.0000 - val_tp: 87.0000
    ## Epoch 17/30
    ## 112/112 - 0s - 3ms/step - fn: 7.0000 - fp: 6110.0000 - loss: 6.0376e-07 - precision: 0.0599 - recall: 0.9823 - tn: 221340.0000 - tp: 389.0000 - val_fn: 8.0000 - val_fp: 1964.0000 - val_loss: 0.1008 - val_precision: 0.0429 - val_recall: 0.9167 - val_tn: 54901.0000 - val_tp: 88.0000
    ## Epoch 18/30
    ## 112/112 - 0s - 3ms/step - fn: 8.0000 - fp: 7499.0000 - loss: 8.7227e-07 - precision: 0.0492 - recall: 0.9798 - tn: 219951.0000 - tp: 388.0000 - val_fn: 8.0000 - val_fp: 1398.0000 - val_loss: 0.0910 - val_precision: 0.0592 - val_recall: 0.9167 - val_tn: 55467.0000 - val_tp: 88.0000
    ## Epoch 19/30
    ## 112/112 - 0s - 3ms/step - fn: 16.0000 - fp: 7767.0000 - loss: 9.3896e-07 - precision: 0.0466 - recall: 0.9596 - tn: 219683.0000 - tp: 380.0000 - val_fn: 10.0000 - val_fp: 1441.0000 - val_loss: 0.0665 - val_precision: 0.0563 - val_recall: 0.8958 - val_tn: 55424.0000 - val_tp: 86.0000
    ## Epoch 20/30
    ## 112/112 - 0s - 3ms/step - fn: 10.0000 - fp: 6682.0000 - loss: 7.0792e-07 - precision: 0.0546 - recall: 0.9747 - tn: 220768.0000 - tp: 386.0000 - val_fn: 9.0000 - val_fp: 2327.0000 - val_loss: 0.0874 - val_precision: 0.0360 - val_recall: 0.9062 - val_tn: 54538.0000 - val_tp: 87.0000
    ## Epoch 21/30
    ## 112/112 - 0s - 3ms/step - fn: 9.0000 - fp: 6684.0000 - loss: 6.5125e-07 - precision: 0.0547 - recall: 0.9773 - tn: 220766.0000 - tp: 387.0000 - val_fn: 10.0000 - val_fp: 1925.0000 - val_loss: 0.1137 - val_precision: 0.0428 - val_recall: 0.8958 - val_tn: 54940.0000 - val_tp: 86.0000
    ## Epoch 22/30
    ## 112/112 - 0s - 3ms/step - fn: 8.0000 - fp: 6129.0000 - loss: 6.1718e-07 - precision: 0.0595 - recall: 0.9798 - tn: 221321.0000 - tp: 388.0000 - val_fn: 10.0000 - val_fp: 961.0000 - val_loss: 0.0497 - val_precision: 0.0821 - val_recall: 0.8958 - val_tn: 55904.0000 - val_tp: 86.0000
    ## Epoch 23/30
    ## 112/112 - 0s - 3ms/step - fn: 4.0000 - fp: 4463.0000 - loss: 3.5531e-07 - precision: 0.0807 - recall: 0.9899 - tn: 222987.0000 - tp: 392.0000 - val_fn: 9.0000 - val_fp: 968.0000 - val_loss: 0.0463 - val_precision: 0.0825 - val_recall: 0.9062 - val_tn: 55897.0000 - val_tp: 87.0000
    ## Epoch 24/30
    ## 112/112 - 0s - 2ms/step - fn: 5.0000 - fp: 4655.0000 - loss: 5.2034e-07 - precision: 0.0775 - recall: 0.9874 - tn: 222795.0000 - tp: 391.0000 - val_fn: 6.0000 - val_fp: 2711.0000 - val_loss: 0.2166 - val_precision: 0.0321 - val_recall: 0.9375 - val_tn: 54154.0000 - val_tp: 90.0000
    ## Epoch 25/30
    ## 112/112 - 0s - 3ms/step - fn: 7.0000 - fp: 5444.0000 - loss: 5.4271e-07 - precision: 0.0667 - recall: 0.9823 - tn: 222006.0000 - tp: 389.0000 - val_fn: 8.0000 - val_fp: 1589.0000 - val_loss: 0.0800 - val_precision: 0.0525 - val_recall: 0.9167 - val_tn: 55276.0000 - val_tp: 88.0000
    ## Epoch 26/30
    ## 112/112 - 0s - 3ms/step - fn: 1.0000 - fp: 2725.0000 - loss: 2.3465e-07 - precision: 0.1266 - recall: 0.9975 - tn: 224725.0000 - tp: 395.0000 - val_fn: 12.0000 - val_fp: 599.0000 - val_loss: 0.0310 - val_precision: 0.1230 - val_recall: 0.8750 - val_tn: 56266.0000 - val_tp: 84.0000
    ## Epoch 27/30
    ## 112/112 - 0s - 3ms/step - fn: 2.0000 - fp: 2568.0000 - loss: 2.0152e-07 - precision: 0.1330 - recall: 0.9949 - tn: 224882.0000 - tp: 394.0000 - val_fn: 10.0000 - val_fp: 600.0000 - val_loss: 0.0307 - val_precision: 0.1254 - val_recall: 0.8958 - val_tn: 56265.0000 - val_tp: 86.0000
    ## Epoch 28/30
    ## 112/112 - 0s - 3ms/step - fn: 2.0000 - fp: 2295.0000 - loss: 2.3650e-07 - precision: 0.1465 - recall: 0.9949 - tn: 225155.0000 - tp: 394.0000 - val_fn: 11.0000 - val_fp: 716.0000 - val_loss: 0.0477 - val_precision: 0.1061 - val_recall: 0.8854 - val_tn: 56149.0000 - val_tp: 85.0000
    ## Epoch 29/30
    ## 112/112 - 0s - 3ms/step - fn: 3.0000 - fp: 3037.0000 - loss: 4.4890e-07 - precision: 0.1146 - recall: 0.9924 - tn: 224413.0000 - tp: 393.0000 - val_fn: 10.0000 - val_fp: 511.0000 - val_loss: 0.0284 - val_precision: 0.1441 - val_recall: 0.8958 - val_tn: 56354.0000 - val_tp: 86.0000
    ## Epoch 30/30
    ## 112/112 - 0s - 3ms/step - fn: 2.0000 - fp: 2169.0000 - loss: 3.0588e-07 - precision: 0.1537 - recall: 0.9949 - tn: 225281.0000 - tp: 394.0000 - val_fn: 9.0000 - val_fp: 1213.0000 - val_loss: 0.1243 - val_precision: 0.0669 - val_recall: 0.9062 - val_tn: 55652.0000 - val_tp: 87.0000

``` r
val_pred <- model %>%
  predict(val_features) %>%
  { as.integer(. > 0.5) }
```

    ## 1781/1781 - 1s - 486us/step

``` r
pred_correct <- val_df$Class == val_pred
cat(sprintf("Validation accuracy: %.2f", mean(pred_correct)))
```

    ## Validation accuracy: 0.98

``` r
fraudulent <- val_df$Class == 1

n_fraudulent_detected <- sum(fraudulent & pred_correct)
n_fraudulent_missed <- sum(fraudulent & !pred_correct)
n_legitimate_flagged <- sum(!fraudulent & !pred_correct)
```

## Conclusions

At the end of training, out of 56,961 validation transactions, we are:

- Correctly identifying 87 of them as fraudulent
- Missing 9 fraudulent transactions
- At the cost of incorrectly flagging 1,213 legitimate transactions

In the real world, one would put an even higher weight on class 1, so as
to reflect that False Negatives are more costly than False Positives.

Next time your credit card gets declined in an online purchase – this is
why.
