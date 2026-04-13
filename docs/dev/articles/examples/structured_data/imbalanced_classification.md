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
    ## 227467    379

``` r
cat(sprintf("Number of positive samples in training data: %i (%.2f%% of total)",
            counts["1"], 100 * counts["1"] / sum(counts)))
```

    ## Number of positive samples in training data: 379 (0.17% of total)

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
    ## 112/112 - 5s - 42ms/step - fn: 48.0000 - fp: 25318.0000 - loss: 2.3771e-06 - precision: 0.0129 - recall: 0.8734 - tn: 202149.0000 - tp: 331.0000 - val_fn: 9.0000 - val_fp: 2117.0000 - val_loss: 0.1573 - val_precision: 0.0468 - val_recall: 0.9204 - val_tn: 54731.0000 - val_tp: 104.0000
    ## Epoch 2/30
    ## 112/112 - 0s - 2ms/step - fn: 28.0000 - fp: 8591.0000 - loss: 1.3124e-06 - precision: 0.0393 - recall: 0.9261 - tn: 218876.0000 - tp: 351.0000 - val_fn: 11.0000 - val_fp: 1035.0000 - val_loss: 0.0623 - val_precision: 0.0897 - val_recall: 0.9027 - val_tn: 55813.0000 - val_tp: 102.0000
    ## Epoch 3/30
    ## 112/112 - 0s - 2ms/step - fn: 28.0000 - fp: 8300.0000 - loss: 1.3011e-06 - precision: 0.0406 - recall: 0.9261 - tn: 219167.0000 - tp: 351.0000 - val_fn: 8.0000 - val_fp: 2601.0000 - val_loss: 0.1574 - val_precision: 0.0388 - val_recall: 0.9292 - val_tn: 54247.0000 - val_tp: 105.0000
    ## Epoch 4/30
    ## 112/112 - 0s - 2ms/step - fn: 18.0000 - fp: 6558.0000 - loss: 1.0091e-06 - precision: 0.0522 - recall: 0.9525 - tn: 220909.0000 - tp: 361.0000 - val_fn: 8.0000 - val_fp: 3487.0000 - val_loss: 0.1890 - val_precision: 0.0292 - val_recall: 0.9292 - val_tn: 53361.0000 - val_tp: 105.0000
    ## Epoch 5/30
    ## 112/112 - 0s - 2ms/step - fn: 18.0000 - fp: 6903.0000 - loss: 8.8053e-07 - precision: 0.0497 - recall: 0.9525 - tn: 220564.0000 - tp: 361.0000 - val_fn: 10.0000 - val_fp: 2243.0000 - val_loss: 0.1160 - val_precision: 0.0439 - val_recall: 0.9115 - val_tn: 54605.0000 - val_tp: 103.0000
    ## Epoch 6/30
    ## 112/112 - 0s - 2ms/step - fn: 14.0000 - fp: 6824.0000 - loss: 7.7964e-07 - precision: 0.0508 - recall: 0.9631 - tn: 220643.0000 - tp: 365.0000 - val_fn: 10.0000 - val_fp: 1042.0000 - val_loss: 0.0738 - val_precision: 0.0900 - val_recall: 0.9115 - val_tn: 55806.0000 - val_tp: 103.0000
    ## Epoch 7/30
    ## 112/112 - 0s - 2ms/step - fn: 16.0000 - fp: 5354.0000 - loss: 6.7227e-07 - precision: 0.0635 - recall: 0.9578 - tn: 222113.0000 - tp: 363.0000 - val_fn: 8.0000 - val_fp: 3329.0000 - val_loss: 0.1349 - val_precision: 0.0306 - val_recall: 0.9292 - val_tn: 53519.0000 - val_tp: 105.0000
    ## Epoch 8/30
    ## 112/112 - 0s - 2ms/step - fn: 13.0000 - fp: 5967.0000 - loss: 5.6794e-07 - precision: 0.0578 - recall: 0.9657 - tn: 221500.0000 - tp: 366.0000 - val_fn: 9.0000 - val_fp: 1792.0000 - val_loss: 0.0743 - val_precision: 0.0549 - val_recall: 0.9204 - val_tn: 55056.0000 - val_tp: 104.0000
    ## Epoch 9/30
    ## 112/112 - 0s - 2ms/step - fn: 11.0000 - fp: 6882.0000 - loss: 6.4828e-07 - precision: 0.0508 - recall: 0.9710 - tn: 220585.0000 - tp: 368.0000 - val_fn: 10.0000 - val_fp: 1946.0000 - val_loss: 0.0915 - val_precision: 0.0503 - val_recall: 0.9115 - val_tn: 54902.0000 - val_tp: 103.0000
    ## Epoch 10/30
    ## 112/112 - 0s - 2ms/step - fn: 13.0000 - fp: 8059.0000 - loss: 6.9696e-07 - precision: 0.0434 - recall: 0.9657 - tn: 219408.0000 - tp: 366.0000 - val_fn: 11.0000 - val_fp: 816.0000 - val_loss: 0.0396 - val_precision: 0.1111 - val_recall: 0.9027 - val_tn: 56032.0000 - val_tp: 102.0000
    ## Epoch 11/30
    ## 112/112 - 0s - 2ms/step - fn: 11.0000 - fp: 5073.0000 - loss: 5.0320e-07 - precision: 0.0676 - recall: 0.9710 - tn: 222394.0000 - tp: 368.0000 - val_fn: 7.0000 - val_fp: 2654.0000 - val_loss: 0.1025 - val_precision: 0.0384 - val_recall: 0.9381 - val_tn: 54194.0000 - val_tp: 106.0000
    ## Epoch 12/30
    ## 112/112 - 0s - 2ms/step - fn: 9.0000 - fp: 6259.0000 - loss: 5.0795e-07 - precision: 0.0558 - recall: 0.9763 - tn: 221208.0000 - tp: 370.0000 - val_fn: 12.0000 - val_fp: 891.0000 - val_loss: 0.0423 - val_precision: 0.1018 - val_recall: 0.8938 - val_tn: 55957.0000 - val_tp: 101.0000
    ## Epoch 13/30
    ## 112/112 - 0s - 2ms/step - fn: 5.0000 - fp: 4864.0000 - loss: 4.3368e-07 - precision: 0.0714 - recall: 0.9868 - tn: 222603.0000 - tp: 374.0000 - val_fn: 11.0000 - val_fp: 774.0000 - val_loss: 0.0361 - val_precision: 0.1164 - val_recall: 0.9027 - val_tn: 56074.0000 - val_tp: 102.0000
    ## Epoch 14/30
    ## 112/112 - 0s - 2ms/step - fn: 6.0000 - fp: 3077.0000 - loss: 3.0232e-07 - precision: 0.1081 - recall: 0.9842 - tn: 224390.0000 - tp: 373.0000 - val_fn: 11.0000 - val_fp: 1308.0000 - val_loss: 0.0607 - val_precision: 0.0723 - val_recall: 0.9027 - val_tn: 55540.0000 - val_tp: 102.0000
    ## Epoch 15/30
    ## 112/112 - 0s - 2ms/step - fn: 8.0000 - fp: 7638.0000 - loss: 6.4551e-07 - precision: 0.0463 - recall: 0.9789 - tn: 219829.0000 - tp: 371.0000 - val_fn: 10.0000 - val_fp: 1557.0000 - val_loss: 0.0681 - val_precision: 0.0620 - val_recall: 0.9115 - val_tn: 55291.0000 - val_tp: 103.0000
    ## Epoch 16/30
    ## 112/112 - 0s - 2ms/step - fn: 10.0000 - fp: 5716.0000 - loss: 5.5256e-07 - precision: 0.0606 - recall: 0.9736 - tn: 221751.0000 - tp: 369.0000 - val_fn: 10.0000 - val_fp: 1420.0000 - val_loss: 0.0590 - val_precision: 0.0676 - val_recall: 0.9115 - val_tn: 55428.0000 - val_tp: 103.0000
    ## Epoch 17/30
    ## 112/112 - 0s - 2ms/step - fn: 2.0000 - fp: 4431.0000 - loss: 3.3036e-07 - precision: 0.0784 - recall: 0.9947 - tn: 223036.0000 - tp: 377.0000 - val_fn: 11.0000 - val_fp: 694.0000 - val_loss: 0.0342 - val_precision: 0.1281 - val_recall: 0.9027 - val_tn: 56154.0000 - val_tp: 102.0000
    ## Epoch 18/30
    ## 112/112 - 0s - 2ms/step - fn: 4.0000 - fp: 3980.0000 - loss: 3.7131e-07 - precision: 0.0861 - recall: 0.9894 - tn: 223487.0000 - tp: 375.0000 - val_fn: 10.0000 - val_fp: 1271.0000 - val_loss: 0.0644 - val_precision: 0.0750 - val_recall: 0.9115 - val_tn: 55577.0000 - val_tp: 103.0000
    ## Epoch 19/30
    ## 112/112 - 0s - 2ms/step - fn: 5.0000 - fp: 5445.0000 - loss: 4.5361e-07 - precision: 0.0643 - recall: 0.9868 - tn: 222022.0000 - tp: 374.0000 - val_fn: 11.0000 - val_fp: 1705.0000 - val_loss: 0.0623 - val_precision: 0.0564 - val_recall: 0.9027 - val_tn: 55143.0000 - val_tp: 102.0000
    ## Epoch 20/30
    ## 112/112 - 0s - 2ms/step - fn: 3.0000 - fp: 4090.0000 - loss: 3.2948e-07 - precision: 0.0842 - recall: 0.9921 - tn: 223377.0000 - tp: 376.0000 - val_fn: 11.0000 - val_fp: 1026.0000 - val_loss: 0.0505 - val_precision: 0.0904 - val_recall: 0.9027 - val_tn: 55822.0000 - val_tp: 102.0000
    ## Epoch 21/30
    ## 112/112 - 0s - 2ms/step - fn: 3.0000 - fp: 3662.0000 - loss: 3.2836e-07 - precision: 0.0931 - recall: 0.9921 - tn: 223805.0000 - tp: 376.0000 - val_fn: 12.0000 - val_fp: 337.0000 - val_loss: 0.0184 - val_precision: 0.2306 - val_recall: 0.8938 - val_tn: 56511.0000 - val_tp: 101.0000
    ## Epoch 22/30
    ## 112/112 - 0s - 2ms/step - fn: 4.0000 - fp: 3436.0000 - loss: 3.0928e-07 - precision: 0.0984 - recall: 0.9894 - tn: 224031.0000 - tp: 375.0000 - val_fn: 12.0000 - val_fp: 1234.0000 - val_loss: 0.0607 - val_precision: 0.0757 - val_recall: 0.8938 - val_tn: 55614.0000 - val_tp: 101.0000
    ## Epoch 23/30
    ## 112/112 - 0s - 2ms/step - fn: 2.0000 - fp: 3295.0000 - loss: 2.6618e-07 - precision: 0.1027 - recall: 0.9947 - tn: 224172.0000 - tp: 377.0000 - val_fn: 10.0000 - val_fp: 1069.0000 - val_loss: 0.0473 - val_precision: 0.0879 - val_recall: 0.9115 - val_tn: 55779.0000 - val_tp: 103.0000
    ## Epoch 24/30
    ## 112/112 - 0s - 2ms/step - fn: 3.0000 - fp: 2650.0000 - loss: 2.6821e-07 - precision: 0.1243 - recall: 0.9921 - tn: 224817.0000 - tp: 376.0000 - val_fn: 10.0000 - val_fp: 2632.0000 - val_loss: 0.2095 - val_precision: 0.0377 - val_recall: 0.9115 - val_tn: 54216.0000 - val_tp: 103.0000
    ## Epoch 25/30
    ## 112/112 - 0s - 2ms/step - fn: 1.0000 - fp: 3692.0000 - loss: 2.9952e-07 - precision: 0.0929 - recall: 0.9974 - tn: 223775.0000 - tp: 378.0000 - val_fn: 13.0000 - val_fp: 738.0000 - val_loss: 0.0359 - val_precision: 0.1193 - val_recall: 0.8850 - val_tn: 56110.0000 - val_tp: 100.0000
    ## Epoch 26/30
    ## 112/112 - 0s - 2ms/step - fn: 1.0000 - fp: 2036.0000 - loss: 1.9598e-07 - precision: 0.1566 - recall: 0.9974 - tn: 225431.0000 - tp: 378.0000 - val_fn: 13.0000 - val_fp: 877.0000 - val_loss: 0.0456 - val_precision: 0.1024 - val_recall: 0.8850 - val_tn: 55971.0000 - val_tp: 100.0000
    ## Epoch 27/30
    ## 112/112 - 0s - 2ms/step - fn: 1.0000 - fp: 2884.0000 - loss: 2.5050e-07 - precision: 0.1159 - recall: 0.9974 - tn: 224583.0000 - tp: 378.0000 - val_fn: 13.0000 - val_fp: 817.0000 - val_loss: 0.0453 - val_precision: 0.1091 - val_recall: 0.8850 - val_tn: 56031.0000 - val_tp: 100.0000
    ## Epoch 28/30
    ## 112/112 - 0s - 2ms/step - fn: 2.0000 - fp: 3546.0000 - loss: 3.1146e-07 - precision: 0.0961 - recall: 0.9947 - tn: 223921.0000 - tp: 377.0000 - val_fn: 12.0000 - val_fp: 1114.0000 - val_loss: 0.0432 - val_precision: 0.0831 - val_recall: 0.8938 - val_tn: 55734.0000 - val_tp: 101.0000
    ## Epoch 29/30
    ## 112/112 - 0s - 2ms/step - fn: 2.0000 - fp: 3590.0000 - loss: 3.7488e-07 - precision: 0.0950 - recall: 0.9947 - tn: 223877.0000 - tp: 377.0000 - val_fn: 12.0000 - val_fp: 1074.0000 - val_loss: 0.0434 - val_precision: 0.0860 - val_recall: 0.8938 - val_tn: 55774.0000 - val_tp: 101.0000
    ## Epoch 30/30
    ## 112/112 - 0s - 2ms/step - fn: 7.0000 - fp: 5566.0000 - loss: 4.3619e-07 - precision: 0.0626 - recall: 0.9815 - tn: 221901.0000 - tp: 372.0000 - val_fn: 12.0000 - val_fp: 634.0000 - val_loss: 0.0307 - val_precision: 0.1374 - val_recall: 0.8938 - val_tn: 56214.0000 - val_tp: 101.0000

``` r
val_pred <- model %>%
  predict(val_features) %>%
  { as.integer(. > 0.5) }
```

    ## 1781/1781 - 1s - 519us/step

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

- Correctly identifying 101 of them as fraudulent
- Missing 12 fraudulent transactions
- At the cost of incorrectly flagging 634 legitimate transactions

In the real world, one would put an even higher weight on class 1, so as
to reflect that False Negatives are more costly than False Positives.

Next time your credit card gets declined in an online purchase – this is
why.
