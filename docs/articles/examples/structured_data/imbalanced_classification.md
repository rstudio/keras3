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
    ## 227460    386

``` r
cat(sprintf("Number of positive samples in training data: %i (%.2f%% of total)",
            counts["1"], 100 * counts["1"] / sum(counts)))
```

    ## Number of positive samples in training data: 386 (0.17% of total)

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
    ## 112/112 - 5s - 42ms/step - fn: 43.0000 - fp: 22739.0000 - loss: 2.1908e-06 - precision: 0.0149 - recall: 0.8886 - tn: 204721.0000 - tp: 343.0000 - val_fn: 13.0000 - val_fp: 1267.0000 - val_loss: 0.0943 - val_precision: 0.0684 - val_recall: 0.8774 - val_tn: 55588.0000 - val_tp: 93.0000
    ## Epoch 2/30
    ## 112/112 - 0s - 2ms/step - fn: 25.0000 - fp: 7620.0000 - loss: 1.1858e-06 - precision: 0.0452 - recall: 0.9352 - tn: 219840.0000 - tp: 361.0000 - val_fn: 13.0000 - val_fp: 1523.0000 - val_loss: 0.0957 - val_precision: 0.0575 - val_recall: 0.8774 - val_tn: 55332.0000 - val_tp: 93.0000
    ## Epoch 3/30
    ## 112/112 - 0s - 2ms/step - fn: 19.0000 - fp: 7148.0000 - loss: 9.3529e-07 - precision: 0.0488 - recall: 0.9508 - tn: 220312.0000 - tp: 367.0000 - val_fn: 12.0000 - val_fp: 2696.0000 - val_loss: 0.1279 - val_precision: 0.0337 - val_recall: 0.8868 - val_tn: 54159.0000 - val_tp: 94.0000
    ## Epoch 4/30
    ## 112/112 - 0s - 2ms/step - fn: 18.0000 - fp: 8001.0000 - loss: 8.6271e-07 - precision: 0.0440 - recall: 0.9534 - tn: 219459.0000 - tp: 368.0000 - val_fn: 12.0000 - val_fp: 2037.0000 - val_loss: 0.0907 - val_precision: 0.0441 - val_recall: 0.8868 - val_tn: 54818.0000 - val_tp: 94.0000
    ## Epoch 5/30
    ## 112/112 - 0s - 2ms/step - fn: 13.0000 - fp: 9192.0000 - loss: 8.3973e-07 - precision: 0.0390 - recall: 0.9663 - tn: 218268.0000 - tp: 373.0000 - val_fn: 13.0000 - val_fp: 2276.0000 - val_loss: 0.1012 - val_precision: 0.0393 - val_recall: 0.8774 - val_tn: 54579.0000 - val_tp: 93.0000
    ## Epoch 6/30
    ## 112/112 - 0s - 2ms/step - fn: 21.0000 - fp: 9720.0000 - loss: 9.8943e-07 - precision: 0.0362 - recall: 0.9456 - tn: 217740.0000 - tp: 365.0000 - val_fn: 10.0000 - val_fp: 4444.0000 - val_loss: 0.2145 - val_precision: 0.0211 - val_recall: 0.9057 - val_tn: 52411.0000 - val_tp: 96.0000
    ## Epoch 7/30
    ## 112/112 - 0s - 2ms/step - fn: 16.0000 - fp: 7151.0000 - loss: 7.5123e-07 - precision: 0.0492 - recall: 0.9585 - tn: 220309.0000 - tp: 370.0000 - val_fn: 11.0000 - val_fp: 4057.0000 - val_loss: 0.1682 - val_precision: 0.0229 - val_recall: 0.8962 - val_tn: 52798.0000 - val_tp: 95.0000
    ## Epoch 8/30
    ## 112/112 - 0s - 2ms/step - fn: 11.0000 - fp: 7965.0000 - loss: 6.1693e-07 - precision: 0.0450 - recall: 0.9715 - tn: 219495.0000 - tp: 375.0000 - val_fn: 13.0000 - val_fp: 1704.0000 - val_loss: 0.0631 - val_precision: 0.0518 - val_recall: 0.8774 - val_tn: 55151.0000 - val_tp: 93.0000
    ## Epoch 9/30
    ## 112/112 - 0s - 2ms/step - fn: 7.0000 - fp: 7583.0000 - loss: 6.3908e-07 - precision: 0.0476 - recall: 0.9819 - tn: 219877.0000 - tp: 379.0000 - val_fn: 14.0000 - val_fp: 708.0000 - val_loss: 0.0417 - val_precision: 0.1150 - val_recall: 0.8679 - val_tn: 56147.0000 - val_tp: 92.0000
    ## Epoch 10/30
    ## 112/112 - 0s - 2ms/step - fn: 10.0000 - fp: 7210.0000 - loss: 5.8323e-07 - precision: 0.0496 - recall: 0.9741 - tn: 220250.0000 - tp: 376.0000 - val_fn: 12.0000 - val_fp: 1305.0000 - val_loss: 0.0553 - val_precision: 0.0672 - val_recall: 0.8868 - val_tn: 55550.0000 - val_tp: 94.0000
    ## Epoch 11/30
    ## 112/112 - 0s - 2ms/step - fn: 5.0000 - fp: 4424.0000 - loss: 3.7127e-07 - precision: 0.0793 - recall: 0.9870 - tn: 223036.0000 - tp: 381.0000 - val_fn: 13.0000 - val_fp: 1361.0000 - val_loss: 0.0668 - val_precision: 0.0640 - val_recall: 0.8774 - val_tn: 55494.0000 - val_tp: 93.0000
    ## Epoch 12/30
    ## 112/112 - 0s - 2ms/step - fn: 7.0000 - fp: 4500.0000 - loss: 3.6361e-07 - precision: 0.0777 - recall: 0.9819 - tn: 222960.0000 - tp: 379.0000 - val_fn: 12.0000 - val_fp: 1510.0000 - val_loss: 0.0569 - val_precision: 0.0586 - val_recall: 0.8868 - val_tn: 55345.0000 - val_tp: 94.0000
    ## Epoch 13/30
    ## 112/112 - 0s - 2ms/step - fn: 7.0000 - fp: 7374.0000 - loss: 5.8414e-07 - precision: 0.0489 - recall: 0.9819 - tn: 220086.0000 - tp: 379.0000 - val_fn: 13.0000 - val_fp: 1630.0000 - val_loss: 0.0585 - val_precision: 0.0540 - val_recall: 0.8774 - val_tn: 55225.0000 - val_tp: 93.0000
    ## Epoch 14/30
    ## 112/112 - 0s - 2ms/step - fn: 3.0000 - fp: 5186.0000 - loss: 3.9560e-07 - precision: 0.0688 - recall: 0.9922 - tn: 222274.0000 - tp: 383.0000 - val_fn: 13.0000 - val_fp: 1262.0000 - val_loss: 0.0541 - val_precision: 0.0686 - val_recall: 0.8774 - val_tn: 55593.0000 - val_tp: 93.0000
    ## Epoch 15/30
    ## 112/112 - 0s - 2ms/step - fn: 14.0000 - fp: 8685.0000 - loss: 1.4041e-06 - precision: 0.0411 - recall: 0.9637 - tn: 218775.0000 - tp: 372.0000 - val_fn: 10.0000 - val_fp: 2944.0000 - val_loss: 0.2771 - val_precision: 0.0316 - val_recall: 0.9057 - val_tn: 53911.0000 - val_tp: 96.0000
    ## Epoch 16/30
    ## 112/112 - 0s - 2ms/step - fn: 8.0000 - fp: 4682.0000 - loss: 5.1692e-07 - precision: 0.0747 - recall: 0.9793 - tn: 222778.0000 - tp: 378.0000 - val_fn: 11.0000 - val_fp: 974.0000 - val_loss: 0.0444 - val_precision: 0.0889 - val_recall: 0.8962 - val_tn: 55881.0000 - val_tp: 95.0000
    ## Epoch 17/30
    ## 112/112 - 0s - 2ms/step - fn: 5.0000 - fp: 3968.0000 - loss: 3.2944e-07 - precision: 0.0876 - recall: 0.9870 - tn: 223492.0000 - tp: 381.0000 - val_fn: 12.0000 - val_fp: 1006.0000 - val_loss: 0.0461 - val_precision: 0.0855 - val_recall: 0.8868 - val_tn: 55849.0000 - val_tp: 94.0000
    ## Epoch 18/30
    ## 112/112 - 0s - 2ms/step - fn: 6.0000 - fp: 4936.0000 - loss: 4.7493e-07 - precision: 0.0715 - recall: 0.9845 - tn: 222524.0000 - tp: 380.0000 - val_fn: 13.0000 - val_fp: 1212.0000 - val_loss: 0.0526 - val_precision: 0.0713 - val_recall: 0.8774 - val_tn: 55643.0000 - val_tp: 93.0000
    ## Epoch 19/30
    ## 112/112 - 0s - 2ms/step - fn: 6.0000 - fp: 5350.0000 - loss: 5.5407e-07 - precision: 0.0663 - recall: 0.9845 - tn: 222110.0000 - tp: 380.0000 - val_fn: 11.0000 - val_fp: 2098.0000 - val_loss: 0.0909 - val_precision: 0.0433 - val_recall: 0.8962 - val_tn: 54757.0000 - val_tp: 95.0000
    ## Epoch 20/30
    ## 112/112 - 0s - 2ms/step - fn: 4.0000 - fp: 4893.0000 - loss: 4.4231e-07 - precision: 0.0724 - recall: 0.9896 - tn: 222567.0000 - tp: 382.0000 - val_fn: 12.0000 - val_fp: 1199.0000 - val_loss: 0.0463 - val_precision: 0.0727 - val_recall: 0.8868 - val_tn: 55656.0000 - val_tp: 94.0000
    ## Epoch 21/30
    ## 112/112 - 0s - 2ms/step - fn: 2.0000 - fp: 3827.0000 - loss: 2.9171e-07 - precision: 0.0912 - recall: 0.9948 - tn: 223633.0000 - tp: 384.0000 - val_fn: 13.0000 - val_fp: 858.0000 - val_loss: 0.0395 - val_precision: 0.0978 - val_recall: 0.8774 - val_tn: 55997.0000 - val_tp: 93.0000
    ## Epoch 22/30
    ## 112/112 - 0s - 2ms/step - fn: 1.0000 - fp: 2273.0000 - loss: 2.0864e-07 - precision: 0.1448 - recall: 0.9974 - tn: 225187.0000 - tp: 385.0000 - val_fn: 12.0000 - val_fp: 1072.0000 - val_loss: 0.0573 - val_precision: 0.0806 - val_recall: 0.8868 - val_tn: 55783.0000 - val_tp: 94.0000
    ## Epoch 23/30
    ## 112/112 - 0s - 2ms/step - fn: 1.0000 - fp: 3199.0000 - loss: 2.5430e-07 - precision: 0.1074 - recall: 0.9974 - tn: 224261.0000 - tp: 385.0000 - val_fn: 14.0000 - val_fp: 482.0000 - val_loss: 0.0251 - val_precision: 0.1603 - val_recall: 0.8679 - val_tn: 56373.0000 - val_tp: 92.0000
    ## Epoch 24/30
    ## 112/112 - 0s - 2ms/step - fn: 2.0000 - fp: 3849.0000 - loss: 3.4936e-07 - precision: 0.0907 - recall: 0.9948 - tn: 223611.0000 - tp: 384.0000 - val_fn: 14.0000 - val_fp: 838.0000 - val_loss: 0.0366 - val_precision: 0.0989 - val_recall: 0.8679 - val_tn: 56017.0000 - val_tp: 92.0000
    ## Epoch 25/30
    ## 112/112 - 0s - 2ms/step - fn: 4.0000 - fp: 3256.0000 - loss: 2.7252e-07 - precision: 0.1050 - recall: 0.9896 - tn: 224204.0000 - tp: 382.0000 - val_fn: 10.0000 - val_fp: 1834.0000 - val_loss: 0.0766 - val_precision: 0.0497 - val_recall: 0.9057 - val_tn: 55021.0000 - val_tp: 96.0000
    ## Epoch 26/30
    ## 112/112 - 0s - 2ms/step - fn: 0.0000e+00 - fp: 2560.0000 - loss: 1.8216e-07 - precision: 0.1310 - recall: 1.0000 - tn: 224900.0000 - tp: 386.0000 - val_fn: 13.0000 - val_fp: 537.0000 - val_loss: 0.0299 - val_precision: 0.1476 - val_recall: 0.8774 - val_tn: 56318.0000 - val_tp: 93.0000
    ## Epoch 27/30
    ## 112/112 - 0s - 2ms/step - fn: 1.0000 - fp: 1652.0000 - loss: 1.4409e-07 - precision: 0.1890 - recall: 0.9974 - tn: 225808.0000 - tp: 385.0000 - val_fn: 16.0000 - val_fp: 272.0000 - val_loss: 0.0193 - val_precision: 0.2486 - val_recall: 0.8491 - val_tn: 56583.0000 - val_tp: 90.0000
    ## Epoch 28/30
    ## 112/112 - 0s - 2ms/step - fn: 4.0000 - fp: 3079.0000 - loss: 5.0763e-07 - precision: 0.1104 - recall: 0.9896 - tn: 224381.0000 - tp: 382.0000 - val_fn: 10.0000 - val_fp: 3034.0000 - val_loss: 0.1621 - val_precision: 0.0307 - val_recall: 0.9057 - val_tn: 53821.0000 - val_tp: 96.0000
    ## Epoch 29/30
    ## 112/112 - 0s - 2ms/step - fn: 6.0000 - fp: 5387.0000 - loss: 5.1741e-07 - precision: 0.0659 - recall: 0.9845 - tn: 222073.0000 - tp: 380.0000 - val_fn: 11.0000 - val_fp: 1705.0000 - val_loss: 0.0777 - val_precision: 0.0528 - val_recall: 0.8962 - val_tn: 55150.0000 - val_tp: 95.0000
    ## Epoch 30/30
    ## 112/112 - 0s - 2ms/step - fn: 4.0000 - fp: 4746.0000 - loss: 4.6045e-07 - precision: 0.0745 - recall: 0.9896 - tn: 222714.0000 - tp: 382.0000 - val_fn: 14.0000 - val_fp: 828.0000 - val_loss: 0.0389 - val_precision: 0.1000 - val_recall: 0.8679 - val_tn: 56027.0000 - val_tp: 92.0000

``` r
val_pred <- model %>%
  predict(val_features) %>%
  { as.integer(. > 0.5) }
```

    ## 1781/1781 - 1s - 570us/step

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

- Correctly identifying 92 of them as fraudulent
- Missing 14 fraudulent transactions
- At the cost of incorrectly flagging 828 legitimate transactions

In the real world, one would put an even higher weight on class 1, so as
to reflect that False Negatives are more costly than False Positives.

Next time your credit card gets declined in an online purchase – this is
why.
