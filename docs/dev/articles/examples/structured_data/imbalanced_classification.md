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
    ## 227448    398

``` r
cat(sprintf("Number of positive samples in training data: %i (%.2f%% of total)",
            counts["1"], 100 * counts["1"] / sum(counts)))
```

    ## Number of positive samples in training data: 398 (0.17% of total)

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
    ## 112/112 - 5s - 42ms/step - fn: 47.0000 - fp: 26573.0000 - loss: 2.2940e-06 - precision: 0.0130 - recall: 0.8819 - tn: 200875.0000 - tp: 351.0000 - val_fn: 10.0000 - val_fp: 2548.0000 - val_loss: 0.1712 - val_precision: 0.0319 - val_recall: 0.8936 - val_tn: 54319.0000 - val_tp: 84.0000
    ## Epoch 2/30
    ## 112/112 - 0s - 2ms/step - fn: 36.0000 - fp: 13857.0000 - loss: 1.8260e-06 - precision: 0.0255 - recall: 0.9095 - tn: 213591.0000 - tp: 362.0000 - val_fn: 12.0000 - val_fp: 669.0000 - val_loss: 0.0792 - val_precision: 0.1092 - val_recall: 0.8723 - val_tn: 56198.0000 - val_tp: 82.0000
    ## Epoch 3/30
    ## 112/112 - 0s - 2ms/step - fn: 25.0000 - fp: 5433.0000 - loss: 1.1157e-06 - precision: 0.0642 - recall: 0.9372 - tn: 222015.0000 - tp: 373.0000 - val_fn: 10.0000 - val_fp: 2759.0000 - val_loss: 0.1609 - val_precision: 0.0295 - val_recall: 0.8936 - val_tn: 54108.0000 - val_tp: 84.0000
    ## Epoch 4/30
    ## 112/112 - 0s - 2ms/step - fn: 20.0000 - fp: 7841.0000 - loss: 1.1623e-06 - precision: 0.0460 - recall: 0.9497 - tn: 219607.0000 - tp: 378.0000 - val_fn: 11.0000 - val_fp: 946.0000 - val_loss: 0.0651 - val_precision: 0.0807 - val_recall: 0.8830 - val_tn: 55921.0000 - val_tp: 83.0000
    ## Epoch 5/30
    ## 112/112 - 0s - 2ms/step - fn: 19.0000 - fp: 4984.0000 - loss: 8.3970e-07 - precision: 0.0707 - recall: 0.9523 - tn: 222464.0000 - tp: 379.0000 - val_fn: 8.0000 - val_fp: 6227.0000 - val_loss: 0.3212 - val_precision: 0.0136 - val_recall: 0.9149 - val_tn: 50640.0000 - val_tp: 86.0000
    ## Epoch 6/30
    ## 112/112 - 0s - 2ms/step - fn: 14.0000 - fp: 7881.0000 - loss: 8.6507e-07 - precision: 0.0465 - recall: 0.9648 - tn: 219567.0000 - tp: 384.0000 - val_fn: 11.0000 - val_fp: 1153.0000 - val_loss: 0.0576 - val_precision: 0.0672 - val_recall: 0.8830 - val_tn: 55714.0000 - val_tp: 83.0000
    ## Epoch 7/30
    ## 112/112 - 0s - 2ms/step - fn: 14.0000 - fp: 5960.0000 - loss: 6.3656e-07 - precision: 0.0605 - recall: 0.9648 - tn: 221488.0000 - tp: 384.0000 - val_fn: 9.0000 - val_fp: 1756.0000 - val_loss: 0.0919 - val_precision: 0.0462 - val_recall: 0.9043 - val_tn: 55111.0000 - val_tp: 85.0000
    ## Epoch 8/30
    ## 112/112 - 0s - 2ms/step - fn: 13.0000 - fp: 7336.0000 - loss: 7.0340e-07 - precision: 0.0499 - recall: 0.9673 - tn: 220112.0000 - tp: 385.0000 - val_fn: 12.0000 - val_fp: 1123.0000 - val_loss: 0.0648 - val_precision: 0.0680 - val_recall: 0.8723 - val_tn: 55744.0000 - val_tp: 82.0000
    ## Epoch 9/30
    ## 112/112 - 0s - 2ms/step - fn: 8.0000 - fp: 5826.0000 - loss: 5.7082e-07 - precision: 0.0627 - recall: 0.9799 - tn: 221622.0000 - tp: 390.0000 - val_fn: 10.0000 - val_fp: 1297.0000 - val_loss: 0.0582 - val_precision: 0.0608 - val_recall: 0.8936 - val_tn: 55570.0000 - val_tp: 84.0000
    ## Epoch 10/30
    ## 112/112 - 0s - 2ms/step - fn: 6.0000 - fp: 5673.0000 - loss: 5.4126e-07 - precision: 0.0646 - recall: 0.9849 - tn: 221775.0000 - tp: 392.0000 - val_fn: 10.0000 - val_fp: 1635.0000 - val_loss: 0.0671 - val_precision: 0.0489 - val_recall: 0.8936 - val_tn: 55232.0000 - val_tp: 84.0000
    ## Epoch 11/30
    ## 112/112 - 0s - 2ms/step - fn: 7.0000 - fp: 5375.0000 - loss: 6.0988e-07 - precision: 0.0678 - recall: 0.9824 - tn: 222073.0000 - tp: 391.0000 - val_fn: 13.0000 - val_fp: 1406.0000 - val_loss: 0.0892 - val_precision: 0.0545 - val_recall: 0.8617 - val_tn: 55461.0000 - val_tp: 81.0000
    ## Epoch 12/30
    ## 112/112 - 0s - 2ms/step - fn: 9.0000 - fp: 7441.0000 - loss: 6.7316e-07 - precision: 0.0497 - recall: 0.9774 - tn: 220007.0000 - tp: 389.0000 - val_fn: 13.0000 - val_fp: 669.0000 - val_loss: 0.0435 - val_precision: 0.1080 - val_recall: 0.8617 - val_tn: 56198.0000 - val_tp: 81.0000
    ## Epoch 13/30
    ## 112/112 - 0s - 2ms/step - fn: 4.0000 - fp: 3489.0000 - loss: 3.3523e-07 - precision: 0.1015 - recall: 0.9899 - tn: 223959.0000 - tp: 394.0000 - val_fn: 13.0000 - val_fp: 617.0000 - val_loss: 0.0337 - val_precision: 0.1160 - val_recall: 0.8617 - val_tn: 56250.0000 - val_tp: 81.0000
    ## Epoch 14/30
    ## 112/112 - 0s - 2ms/step - fn: 10.0000 - fp: 5190.0000 - loss: 4.8929e-07 - precision: 0.0696 - recall: 0.9749 - tn: 222258.0000 - tp: 388.0000 - val_fn: 9.0000 - val_fp: 2192.0000 - val_loss: 0.1018 - val_precision: 0.0373 - val_recall: 0.9043 - val_tn: 54675.0000 - val_tp: 85.0000
    ## Epoch 15/30
    ## 112/112 - 0s - 2ms/step - fn: 6.0000 - fp: 5348.0000 - loss: 4.3230e-07 - precision: 0.0683 - recall: 0.9849 - tn: 222100.0000 - tp: 392.0000 - val_fn: 11.0000 - val_fp: 3056.0000 - val_loss: 0.1279 - val_precision: 0.0264 - val_recall: 0.8830 - val_tn: 53811.0000 - val_tp: 83.0000
    ## Epoch 16/30
    ## 112/112 - 0s - 2ms/step - fn: 5.0000 - fp: 4834.0000 - loss: 3.7224e-07 - precision: 0.0752 - recall: 0.9874 - tn: 222614.0000 - tp: 393.0000 - val_fn: 12.0000 - val_fp: 617.0000 - val_loss: 0.0322 - val_precision: 0.1173 - val_recall: 0.8723 - val_tn: 56250.0000 - val_tp: 82.0000
    ## Epoch 17/30
    ## 112/112 - 0s - 2ms/step - fn: 6.0000 - fp: 5150.0000 - loss: 4.2131e-07 - precision: 0.0707 - recall: 0.9849 - tn: 222298.0000 - tp: 392.0000 - val_fn: 13.0000 - val_fp: 620.0000 - val_loss: 0.0363 - val_precision: 0.1155 - val_recall: 0.8617 - val_tn: 56247.0000 - val_tp: 81.0000
    ## Epoch 18/30
    ## 112/112 - 0s - 2ms/step - fn: 5.0000 - fp: 3020.0000 - loss: 3.0311e-07 - precision: 0.1151 - recall: 0.9874 - tn: 224428.0000 - tp: 393.0000 - val_fn: 11.0000 - val_fp: 593.0000 - val_loss: 0.0332 - val_precision: 0.1228 - val_recall: 0.8830 - val_tn: 56274.0000 - val_tp: 83.0000
    ## Epoch 19/30
    ## 112/112 - 0s - 2ms/step - fn: 2.0000 - fp: 3154.0000 - loss: 2.5504e-07 - precision: 0.1115 - recall: 0.9950 - tn: 224294.0000 - tp: 396.0000 - val_fn: 12.0000 - val_fp: 863.0000 - val_loss: 0.0457 - val_precision: 0.0868 - val_recall: 0.8723 - val_tn: 56004.0000 - val_tp: 82.0000
    ## Epoch 20/30
    ## 112/112 - 0s - 2ms/step - fn: 4.0000 - fp: 5315.0000 - loss: 4.6083e-07 - precision: 0.0690 - recall: 0.9899 - tn: 222133.0000 - tp: 394.0000 - val_fn: 11.0000 - val_fp: 867.0000 - val_loss: 0.0490 - val_precision: 0.0874 - val_recall: 0.8830 - val_tn: 56000.0000 - val_tp: 83.0000
    ## Epoch 21/30
    ## 112/112 - 0s - 2ms/step - fn: 4.0000 - fp: 4040.0000 - loss: 4.3505e-07 - precision: 0.0889 - recall: 0.9899 - tn: 223408.0000 - tp: 394.0000 - val_fn: 11.0000 - val_fp: 1104.0000 - val_loss: 0.0490 - val_precision: 0.0699 - val_recall: 0.8830 - val_tn: 55763.0000 - val_tp: 83.0000
    ## Epoch 22/30
    ## 112/112 - 0s - 2ms/step - fn: 6.0000 - fp: 6280.0000 - loss: 5.3667e-07 - precision: 0.0588 - recall: 0.9849 - tn: 221168.0000 - tp: 392.0000 - val_fn: 11.0000 - val_fp: 573.0000 - val_loss: 0.0306 - val_precision: 0.1265 - val_recall: 0.8830 - val_tn: 56294.0000 - val_tp: 83.0000
    ## Epoch 23/30
    ## 112/112 - 0s - 2ms/step - fn: 4.0000 - fp: 3863.0000 - loss: 3.9085e-07 - precision: 0.0926 - recall: 0.9899 - tn: 223585.0000 - tp: 394.0000 - val_fn: 10.0000 - val_fp: 1141.0000 - val_loss: 0.0498 - val_precision: 0.0686 - val_recall: 0.8936 - val_tn: 55726.0000 - val_tp: 84.0000
    ## Epoch 24/30
    ## 112/112 - 0s - 2ms/step - fn: 6.0000 - fp: 4593.0000 - loss: 4.3019e-07 - precision: 0.0786 - recall: 0.9849 - tn: 222855.0000 - tp: 392.0000 - val_fn: 9.0000 - val_fp: 1706.0000 - val_loss: 0.0643 - val_precision: 0.0475 - val_recall: 0.9043 - val_tn: 55161.0000 - val_tp: 85.0000
    ## Epoch 25/30
    ## 112/112 - 0s - 2ms/step - fn: 3.0000 - fp: 4266.0000 - loss: 3.2117e-07 - precision: 0.0847 - recall: 0.9925 - tn: 223182.0000 - tp: 395.0000 - val_fn: 10.0000 - val_fp: 1651.0000 - val_loss: 0.0764 - val_precision: 0.0484 - val_recall: 0.8936 - val_tn: 55216.0000 - val_tp: 84.0000
    ## Epoch 26/30
    ## 112/112 - 0s - 2ms/step - fn: 2.0000 - fp: 3221.0000 - loss: 2.9923e-07 - precision: 0.1095 - recall: 0.9950 - tn: 224227.0000 - tp: 396.0000 - val_fn: 9.0000 - val_fp: 1494.0000 - val_loss: 0.0977 - val_precision: 0.0538 - val_recall: 0.9043 - val_tn: 55373.0000 - val_tp: 85.0000
    ## Epoch 27/30
    ## 112/112 - 0s - 2ms/step - fn: 0.0000e+00 - fp: 3195.0000 - loss: 2.6797e-07 - precision: 0.1108 - recall: 1.0000 - tn: 224253.0000 - tp: 398.0000 - val_fn: 12.0000 - val_fp: 470.0000 - val_loss: 0.0303 - val_precision: 0.1486 - val_recall: 0.8723 - val_tn: 56397.0000 - val_tp: 82.0000
    ## Epoch 28/30
    ## 112/112 - 0s - 2ms/step - fn: 3.0000 - fp: 2700.0000 - loss: 2.6402e-07 - precision: 0.1276 - recall: 0.9925 - tn: 224748.0000 - tp: 395.0000 - val_fn: 13.0000 - val_fp: 711.0000 - val_loss: 0.0381 - val_precision: 0.1023 - val_recall: 0.8617 - val_tn: 56156.0000 - val_tp: 81.0000
    ## Epoch 29/30
    ## 112/112 - 0s - 2ms/step - fn: 1.0000 - fp: 1625.0000 - loss: 1.7383e-07 - precision: 0.1963 - recall: 0.9975 - tn: 225823.0000 - tp: 397.0000 - val_fn: 13.0000 - val_fp: 543.0000 - val_loss: 0.0313 - val_precision: 0.1298 - val_recall: 0.8617 - val_tn: 56324.0000 - val_tp: 81.0000
    ## Epoch 30/30
    ## 112/112 - 0s - 2ms/step - fn: 0.0000e+00 - fp: 1564.0000 - loss: 1.3238e-07 - precision: 0.2029 - recall: 1.0000 - tn: 225884.0000 - tp: 398.0000 - val_fn: 13.0000 - val_fp: 309.0000 - val_loss: 0.0218 - val_precision: 0.2077 - val_recall: 0.8617 - val_tn: 56558.0000 - val_tp: 81.0000

``` r
val_pred <- model %>%
  predict(val_features) %>%
  { as.integer(. > 0.5) }
```

    ## 1781/1781 - 1s - 556us/step

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

- Correctly identifying 81 of them as fraudulent
- Missing 13 fraudulent transactions
- At the cost of incorrectly flagging 309 legitimate transactions

In the real world, one would put an even higher weight on class 1, so as
to reflect that False Negatives are more costly than False Positives.

Next time your credit card gets declined in an online purchase – this is
why.
