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
    ## 227462    384

``` r
cat(sprintf("Number of positive samples in training data: %i (%.2f%% of total)",
            counts["1"], 100 * counts["1"] / sum(counts)))
```

    ## Number of positive samples in training data: 384 (0.17% of total)

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
    ## 112/112 - 2s - 22ms/step - fn: 39.0000 - fp: 31543.0000 - loss: 2.2524e-06 - precision: 0.0108 - recall: 0.8984 - tn: 195919.0000 - tp: 345.0000 - val_fn: 13.0000 - val_fp: 1471.0000 - val_loss: 0.1056 - val_precision: 0.0607 - val_recall: 0.8796 - val_tn: 55382.0000 - val_tp: 95.0000
    ## Epoch 2/30
    ## 112/112 - 0s - 3ms/step - fn: 31.0000 - fp: 8383.0000 - loss: 1.3401e-06 - precision: 0.0404 - recall: 0.9193 - tn: 219079.0000 - tp: 353.0000 - val_fn: 12.0000 - val_fp: 1255.0000 - val_loss: 0.1056 - val_precision: 0.0711 - val_recall: 0.8889 - val_tn: 55598.0000 - val_tp: 96.0000
    ## Epoch 3/30
    ## 112/112 - 0s - 2ms/step - fn: 26.0000 - fp: 7147.0000 - loss: 1.0722e-06 - precision: 0.0477 - recall: 0.9323 - tn: 220315.0000 - tp: 358.0000 - val_fn: 11.0000 - val_fp: 1142.0000 - val_loss: 0.0842 - val_precision: 0.0783 - val_recall: 0.8981 - val_tn: 55711.0000 - val_tp: 97.0000
    ## Epoch 4/30
    ## 112/112 - 0s - 3ms/step - fn: 20.0000 - fp: 7168.0000 - loss: 9.0571e-07 - precision: 0.0483 - recall: 0.9479 - tn: 220294.0000 - tp: 364.0000 - val_fn: 9.0000 - val_fp: 2771.0000 - val_loss: 0.1210 - val_precision: 0.0345 - val_recall: 0.9167 - val_tn: 54082.0000 - val_tp: 99.0000
    ## Epoch 5/30
    ## 112/112 - 0s - 3ms/step - fn: 21.0000 - fp: 9325.0000 - loss: 1.0877e-06 - precision: 0.0375 - recall: 0.9453 - tn: 218137.0000 - tp: 363.0000 - val_fn: 10.0000 - val_fp: 3146.0000 - val_loss: 0.1408 - val_precision: 0.0302 - val_recall: 0.9074 - val_tn: 53707.0000 - val_tp: 98.0000
    ## Epoch 6/30
    ## 112/112 - 0s - 3ms/step - fn: 23.0000 - fp: 7062.0000 - loss: 8.4578e-07 - precision: 0.0486 - recall: 0.9401 - tn: 220400.0000 - tp: 361.0000 - val_fn: 10.0000 - val_fp: 1826.0000 - val_loss: 0.0834 - val_precision: 0.0509 - val_recall: 0.9074 - val_tn: 55027.0000 - val_tp: 98.0000
    ## Epoch 7/30
    ## 112/112 - 0s - 3ms/step - fn: 15.0000 - fp: 6498.0000 - loss: 7.1603e-07 - precision: 0.0537 - recall: 0.9609 - tn: 220964.0000 - tp: 369.0000 - val_fn: 13.0000 - val_fp: 1234.0000 - val_loss: 0.0524 - val_precision: 0.0715 - val_recall: 0.8796 - val_tn: 55619.0000 - val_tp: 95.0000
    ## Epoch 8/30
    ## 112/112 - 0s - 3ms/step - fn: 9.0000 - fp: 7383.0000 - loss: 6.7657e-07 - precision: 0.0483 - recall: 0.9766 - tn: 220079.0000 - tp: 375.0000 - val_fn: 14.0000 - val_fp: 1324.0000 - val_loss: 0.0672 - val_precision: 0.0663 - val_recall: 0.8704 - val_tn: 55529.0000 - val_tp: 94.0000
    ## Epoch 9/30
    ## 112/112 - 0s - 3ms/step - fn: 13.0000 - fp: 7117.0000 - loss: 6.7631e-07 - precision: 0.0495 - recall: 0.9661 - tn: 220345.0000 - tp: 371.0000 - val_fn: 7.0000 - val_fp: 4334.0000 - val_loss: 0.2544 - val_precision: 0.0228 - val_recall: 0.9352 - val_tn: 52519.0000 - val_tp: 101.0000
    ## Epoch 10/30
    ## 112/112 - 0s - 3ms/step - fn: 12.0000 - fp: 7891.0000 - loss: 6.6250e-07 - precision: 0.0450 - recall: 0.9688 - tn: 219571.0000 - tp: 372.0000 - val_fn: 14.0000 - val_fp: 971.0000 - val_loss: 0.0374 - val_precision: 0.0883 - val_recall: 0.8704 - val_tn: 55882.0000 - val_tp: 94.0000
    ## Epoch 11/30
    ## 112/112 - 0s - 3ms/step - fn: 10.0000 - fp: 6939.0000 - loss: 6.1831e-07 - precision: 0.0511 - recall: 0.9740 - tn: 220523.0000 - tp: 374.0000 - val_fn: 14.0000 - val_fp: 1550.0000 - val_loss: 0.0725 - val_precision: 0.0572 - val_recall: 0.8704 - val_tn: 55303.0000 - val_tp: 94.0000
    ## Epoch 12/30
    ## 112/112 - 0s - 3ms/step - fn: 8.0000 - fp: 6447.0000 - loss: 5.1876e-07 - precision: 0.0551 - recall: 0.9792 - tn: 221015.0000 - tp: 376.0000 - val_fn: 10.0000 - val_fp: 3785.0000 - val_loss: 0.1702 - val_precision: 0.0252 - val_recall: 0.9074 - val_tn: 53068.0000 - val_tp: 98.0000
    ## Epoch 13/30
    ## 112/112 - 0s - 3ms/step - fn: 3.0000 - fp: 7020.0000 - loss: 5.3519e-07 - precision: 0.0515 - recall: 0.9922 - tn: 220442.0000 - tp: 381.0000 - val_fn: 13.0000 - val_fp: 2107.0000 - val_loss: 0.0997 - val_precision: 0.0431 - val_recall: 0.8796 - val_tn: 54746.0000 - val_tp: 95.0000
    ## Epoch 14/30
    ## 112/112 - 0s - 3ms/step - fn: 5.0000 - fp: 6306.0000 - loss: 4.7681e-07 - precision: 0.0567 - recall: 0.9870 - tn: 221156.0000 - tp: 379.0000 - val_fn: 12.0000 - val_fp: 1704.0000 - val_loss: 0.0676 - val_precision: 0.0533 - val_recall: 0.8889 - val_tn: 55149.0000 - val_tp: 96.0000
    ## Epoch 15/30
    ## 112/112 - 0s - 3ms/step - fn: 5.0000 - fp: 4847.0000 - loss: 3.4897e-07 - precision: 0.0725 - recall: 0.9870 - tn: 222615.0000 - tp: 379.0000 - val_fn: 14.0000 - val_fp: 557.0000 - val_loss: 0.0286 - val_precision: 0.1444 - val_recall: 0.8704 - val_tn: 56296.0000 - val_tp: 94.0000
    ## Epoch 16/30
    ## 112/112 - 0s - 3ms/step - fn: 5.0000 - fp: 3180.0000 - loss: 2.9365e-07 - precision: 0.1065 - recall: 0.9870 - tn: 224282.0000 - tp: 379.0000 - val_fn: 13.0000 - val_fp: 786.0000 - val_loss: 0.0347 - val_precision: 0.1078 - val_recall: 0.8796 - val_tn: 56067.0000 - val_tp: 95.0000
    ## Epoch 17/30
    ## 112/112 - 0s - 3ms/step - fn: 4.0000 - fp: 6478.0000 - loss: 5.4673e-07 - precision: 0.0554 - recall: 0.9896 - tn: 220984.0000 - tp: 380.0000 - val_fn: 13.0000 - val_fp: 2107.0000 - val_loss: 0.0727 - val_precision: 0.0431 - val_recall: 0.8796 - val_tn: 54746.0000 - val_tp: 95.0000
    ## Epoch 18/30
    ## 112/112 - 0s - 3ms/step - fn: 6.0000 - fp: 5766.0000 - loss: 4.1548e-07 - precision: 0.0615 - recall: 0.9844 - tn: 221696.0000 - tp: 378.0000 - val_fn: 13.0000 - val_fp: 1012.0000 - val_loss: 0.0433 - val_precision: 0.0858 - val_recall: 0.8796 - val_tn: 55841.0000 - val_tp: 95.0000
    ## Epoch 19/30
    ## 112/112 - 0s - 3ms/step - fn: 5.0000 - fp: 4289.0000 - loss: 3.3983e-07 - precision: 0.0812 - recall: 0.9870 - tn: 223173.0000 - tp: 379.0000 - val_fn: 13.0000 - val_fp: 1058.0000 - val_loss: 0.0434 - val_precision: 0.0824 - val_recall: 0.8796 - val_tn: 55795.0000 - val_tp: 95.0000
    ## Epoch 20/30
    ## 112/112 - 0s - 3ms/step - fn: 3.0000 - fp: 2751.0000 - loss: 2.3358e-07 - precision: 0.1216 - recall: 0.9922 - tn: 224711.0000 - tp: 381.0000 - val_fn: 11.0000 - val_fp: 1771.0000 - val_loss: 0.0955 - val_precision: 0.0519 - val_recall: 0.8981 - val_tn: 55082.0000 - val_tp: 97.0000
    ## Epoch 21/30
    ## 112/112 - 0s - 3ms/step - fn: 1.0000 - fp: 3949.0000 - loss: 3.3070e-07 - precision: 0.0884 - recall: 0.9974 - tn: 223513.0000 - tp: 383.0000 - val_fn: 14.0000 - val_fp: 1389.0000 - val_loss: 0.0551 - val_precision: 0.0634 - val_recall: 0.8704 - val_tn: 55464.0000 - val_tp: 94.0000
    ## Epoch 22/30
    ## 112/112 - 0s - 3ms/step - fn: 3.0000 - fp: 3552.0000 - loss: 3.3962e-07 - precision: 0.0969 - recall: 0.9922 - tn: 223910.0000 - tp: 381.0000 - val_fn: 13.0000 - val_fp: 1122.0000 - val_loss: 0.0491 - val_precision: 0.0781 - val_recall: 0.8796 - val_tn: 55731.0000 - val_tp: 95.0000
    ## Epoch 23/30
    ## 112/112 - 0s - 3ms/step - fn: 2.0000 - fp: 3387.0000 - loss: 2.4427e-07 - precision: 0.1014 - recall: 0.9948 - tn: 224075.0000 - tp: 382.0000 - val_fn: 14.0000 - val_fp: 654.0000 - val_loss: 0.0307 - val_precision: 0.1257 - val_recall: 0.8704 - val_tn: 56199.0000 - val_tp: 94.0000
    ## Epoch 24/30
    ## 112/112 - 0s - 3ms/step - fn: 1.0000 - fp: 2553.0000 - loss: 1.9860e-07 - precision: 0.1304 - recall: 0.9974 - tn: 224909.0000 - tp: 383.0000 - val_fn: 14.0000 - val_fp: 483.0000 - val_loss: 0.0258 - val_precision: 0.1629 - val_recall: 0.8704 - val_tn: 56370.0000 - val_tp: 94.0000
    ## Epoch 25/30
    ## 112/112 - 0s - 3ms/step - fn: 1.0000 - fp: 1882.0000 - loss: 1.5506e-07 - precision: 0.1691 - recall: 0.9974 - tn: 225580.0000 - tp: 383.0000 - val_fn: 14.0000 - val_fp: 295.0000 - val_loss: 0.0196 - val_precision: 0.2416 - val_recall: 0.8704 - val_tn: 56558.0000 - val_tp: 94.0000
    ## Epoch 26/30
    ## 112/112 - 0s - 3ms/step - fn: 2.0000 - fp: 2071.0000 - loss: 2.0557e-07 - precision: 0.1557 - recall: 0.9948 - tn: 225391.0000 - tp: 382.0000 - val_fn: 13.0000 - val_fp: 1811.0000 - val_loss: 0.1638 - val_precision: 0.0498 - val_recall: 0.8796 - val_tn: 55042.0000 - val_tp: 95.0000
    ## Epoch 27/30
    ## 112/112 - 0s - 3ms/step - fn: 6.0000 - fp: 4846.0000 - loss: 4.6646e-07 - precision: 0.0724 - recall: 0.9844 - tn: 222616.0000 - tp: 378.0000 - val_fn: 13.0000 - val_fp: 3399.0000 - val_loss: 0.1693 - val_precision: 0.0272 - val_recall: 0.8796 - val_tn: 53454.0000 - val_tp: 95.0000
    ## Epoch 28/30
    ## 112/112 - 0s - 3ms/step - fn: 3.0000 - fp: 4453.0000 - loss: 3.7956e-07 - precision: 0.0788 - recall: 0.9922 - tn: 223009.0000 - tp: 381.0000 - val_fn: 14.0000 - val_fp: 707.0000 - val_loss: 0.0390 - val_precision: 0.1174 - val_recall: 0.8704 - val_tn: 56146.0000 - val_tp: 94.0000
    ## Epoch 29/30
    ## 112/112 - 0s - 3ms/step - fn: 0.0000e+00 - fp: 2335.0000 - loss: 1.9262e-07 - precision: 0.1412 - recall: 1.0000 - tn: 225127.0000 - tp: 384.0000 - val_fn: 14.0000 - val_fp: 502.0000 - val_loss: 0.0283 - val_precision: 0.1577 - val_recall: 0.8704 - val_tn: 56351.0000 - val_tp: 94.0000
    ## Epoch 30/30
    ## 112/112 - 0s - 3ms/step - fn: 2.0000 - fp: 2250.0000 - loss: 2.5369e-07 - precision: 0.1451 - recall: 0.9948 - tn: 225212.0000 - tp: 382.0000 - val_fn: 14.0000 - val_fp: 858.0000 - val_loss: 0.0874 - val_precision: 0.0987 - val_recall: 0.8704 - val_tn: 55995.0000 - val_tp: 94.0000

``` r
val_pred <- model %>%
  predict(val_features) %>%
  { as.integer(. > 0.5) }
```

    ## 1781/1781 - 1s - 457us/step

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

- Correctly identifying 94 of them as fraudulent
- Missing 14 fraudulent transactions
- At the cost of incorrectly flagging 858 legitimate transactions

In the real world, one would put an even higher weight on class 1, so as
to reflect that False Negatives are more costly than False Positives.

Next time your credit card gets declined in an online purchase – this is
why.
