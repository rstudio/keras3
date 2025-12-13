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
    ## 227464    382

``` r
cat(sprintf("Number of positive samples in training data: %i (%.2f%% of total)",
            counts["1"], 100 * counts["1"] / sum(counts)))
```

    ## Number of positive samples in training data: 382 (0.17% of total)

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
    ## 112/112 - 4s - 37ms/step - fn: 36.0000 - fp: 29717.0000 - loss: 2.3629e-06 - precision: 0.0115 - recall: 0.9058 - tn: 197747.0000 - tp: 346.0000 - val_fn: 12.0000 - val_fp: 1039.0000 - val_loss: 0.0984 - val_precision: 0.0862 - val_recall: 0.8909 - val_tn: 55812.0000 - val_tp: 98.0000
    ## Epoch 2/30
    ## 112/112 - 1s - 9ms/step - fn: 30.0000 - fp: 6694.0000 - loss: 1.3508e-06 - precision: 0.0500 - recall: 0.9215 - tn: 220770.0000 - tp: 352.0000 - val_fn: 8.0000 - val_fp: 2418.0000 - val_loss: 0.1390 - val_precision: 0.0405 - val_recall: 0.9273 - val_tn: 54433.0000 - val_tp: 102.0000
    ## Epoch 3/30
    ## 112/112 - 0s - 2ms/step - fn: 27.0000 - fp: 9781.0000 - loss: 1.3090e-06 - precision: 0.0350 - recall: 0.9293 - tn: 217683.0000 - tp: 355.0000 - val_fn: 10.0000 - val_fp: 984.0000 - val_loss: 0.1063 - val_precision: 0.0923 - val_recall: 0.9091 - val_tn: 55867.0000 - val_tp: 100.0000
    ## Epoch 4/30
    ## 112/112 - 0s - 2ms/step - fn: 32.0000 - fp: 6894.0000 - loss: 1.1099e-06 - precision: 0.0483 - recall: 0.9162 - tn: 220570.0000 - tp: 350.0000 - val_fn: 9.0000 - val_fp: 1557.0000 - val_loss: 0.0857 - val_precision: 0.0609 - val_recall: 0.9182 - val_tn: 55294.0000 - val_tp: 101.0000
    ## Epoch 5/30
    ## 112/112 - 0s - 2ms/step - fn: 23.0000 - fp: 8892.0000 - loss: 9.2992e-07 - precision: 0.0388 - recall: 0.9398 - tn: 218572.0000 - tp: 359.0000 - val_fn: 7.0000 - val_fp: 1492.0000 - val_loss: 0.0665 - val_precision: 0.0646 - val_recall: 0.9364 - val_tn: 55359.0000 - val_tp: 103.0000
    ## Epoch 6/30
    ## 112/112 - 0s - 2ms/step - fn: 18.0000 - fp: 8209.0000 - loss: 7.7549e-07 - precision: 0.0425 - recall: 0.9529 - tn: 219255.0000 - tp: 364.0000 - val_fn: 7.0000 - val_fp: 2749.0000 - val_loss: 0.1010 - val_precision: 0.0361 - val_recall: 0.9364 - val_tn: 54102.0000 - val_tp: 103.0000
    ## Epoch 7/30
    ## 112/112 - 0s - 2ms/step - fn: 15.0000 - fp: 7189.0000 - loss: 6.9032e-07 - precision: 0.0486 - recall: 0.9607 - tn: 220275.0000 - tp: 367.0000 - val_fn: 12.0000 - val_fp: 934.0000 - val_loss: 0.0539 - val_precision: 0.0950 - val_recall: 0.8909 - val_tn: 55917.0000 - val_tp: 98.0000
    ## Epoch 8/30
    ## 112/112 - 0s - 2ms/step - fn: 18.0000 - fp: 7759.0000 - loss: 8.2432e-07 - precision: 0.0448 - recall: 0.9529 - tn: 219705.0000 - tp: 364.0000 - val_fn: 10.0000 - val_fp: 1481.0000 - val_loss: 0.0777 - val_precision: 0.0633 - val_recall: 0.9091 - val_tn: 55370.0000 - val_tp: 100.0000
    ## Epoch 9/30
    ## 112/112 - 0s - 2ms/step - fn: 16.0000 - fp: 7565.0000 - loss: 7.2956e-07 - precision: 0.0461 - recall: 0.9581 - tn: 219899.0000 - tp: 366.0000 - val_fn: 8.0000 - val_fp: 2075.0000 - val_loss: 0.0718 - val_precision: 0.0469 - val_recall: 0.9273 - val_tn: 54776.0000 - val_tp: 102.0000
    ## Epoch 10/30
    ## 112/112 - 0s - 2ms/step - fn: 10.0000 - fp: 7374.0000 - loss: 5.9522e-07 - precision: 0.0480 - recall: 0.9738 - tn: 220090.0000 - tp: 372.0000 - val_fn: 11.0000 - val_fp: 995.0000 - val_loss: 0.0425 - val_precision: 0.0905 - val_recall: 0.9000 - val_tn: 55856.0000 - val_tp: 99.0000
    ## Epoch 11/30
    ## 112/112 - 0s - 2ms/step - fn: 10.0000 - fp: 6586.0000 - loss: 6.2194e-07 - precision: 0.0535 - recall: 0.9738 - tn: 220878.0000 - tp: 372.0000 - val_fn: 9.0000 - val_fp: 5097.0000 - val_loss: 0.2103 - val_precision: 0.0194 - val_recall: 0.9182 - val_tn: 51754.0000 - val_tp: 101.0000
    ## Epoch 12/30
    ## 112/112 - 0s - 2ms/step - fn: 6.0000 - fp: 8285.0000 - loss: 6.0991e-07 - precision: 0.0434 - recall: 0.9843 - tn: 219179.0000 - tp: 376.0000 - val_fn: 9.0000 - val_fp: 1926.0000 - val_loss: 0.0781 - val_precision: 0.0498 - val_recall: 0.9182 - val_tn: 54925.0000 - val_tp: 101.0000
    ## Epoch 13/30
    ## 112/112 - 0s - 2ms/step - fn: 7.0000 - fp: 6864.0000 - loss: 5.7087e-07 - precision: 0.0518 - recall: 0.9817 - tn: 220600.0000 - tp: 375.0000 - val_fn: 8.0000 - val_fp: 1900.0000 - val_loss: 0.0837 - val_precision: 0.0509 - val_recall: 0.9273 - val_tn: 54951.0000 - val_tp: 102.0000
    ## Epoch 14/30
    ## 112/112 - 0s - 2ms/step - fn: 7.0000 - fp: 7663.0000 - loss: 5.7664e-07 - precision: 0.0467 - recall: 0.9817 - tn: 219801.0000 - tp: 375.0000 - val_fn: 8.0000 - val_fp: 1540.0000 - val_loss: 0.0656 - val_precision: 0.0621 - val_recall: 0.9273 - val_tn: 55311.0000 - val_tp: 102.0000
    ## Epoch 15/30
    ## 112/112 - 0s - 2ms/step - fn: 6.0000 - fp: 4607.0000 - loss: 3.9004e-07 - precision: 0.0755 - recall: 0.9843 - tn: 222857.0000 - tp: 376.0000 - val_fn: 11.0000 - val_fp: 2546.0000 - val_loss: 0.0994 - val_precision: 0.0374 - val_recall: 0.9000 - val_tn: 54305.0000 - val_tp: 99.0000
    ## Epoch 16/30
    ## 112/112 - 0s - 2ms/step - fn: 6.0000 - fp: 6195.0000 - loss: 5.1274e-07 - precision: 0.0572 - recall: 0.9843 - tn: 221269.0000 - tp: 376.0000 - val_fn: 9.0000 - val_fp: 1652.0000 - val_loss: 0.0693 - val_precision: 0.0576 - val_recall: 0.9182 - val_tn: 55199.0000 - val_tp: 101.0000
    ## Epoch 17/30
    ## 112/112 - 0s - 2ms/step - fn: 6.0000 - fp: 5367.0000 - loss: 4.6364e-07 - precision: 0.0655 - recall: 0.9843 - tn: 222097.0000 - tp: 376.0000 - val_fn: 10.0000 - val_fp: 1378.0000 - val_loss: 0.0575 - val_precision: 0.0677 - val_recall: 0.9091 - val_tn: 55473.0000 - val_tp: 100.0000
    ## Epoch 18/30
    ## 112/112 - 0s - 2ms/step - fn: 2.0000 - fp: 3696.0000 - loss: 2.9266e-07 - precision: 0.0932 - recall: 0.9948 - tn: 223768.0000 - tp: 380.0000 - val_fn: 11.0000 - val_fp: 824.0000 - val_loss: 0.0375 - val_precision: 0.1073 - val_recall: 0.9000 - val_tn: 56027.0000 - val_tp: 99.0000
    ## Epoch 19/30
    ## 112/112 - 0s - 2ms/step - fn: 3.0000 - fp: 3638.0000 - loss: 3.6945e-07 - precision: 0.0943 - recall: 0.9921 - tn: 223826.0000 - tp: 379.0000 - val_fn: 4.0000 - val_fp: 4639.0000 - val_loss: 0.5475 - val_precision: 0.0223 - val_recall: 0.9636 - val_tn: 52212.0000 - val_tp: 106.0000
    ## Epoch 20/30
    ## 112/112 - 0s - 2ms/step - fn: 26.0000 - fp: 11743.0000 - loss: 4.2168e-06 - precision: 0.0294 - recall: 0.9319 - tn: 215721.0000 - tp: 356.0000 - val_fn: 104.0000 - val_fp: 395.0000 - val_loss: 0.0507 - val_precision: 0.0150 - val_recall: 0.0545 - val_tn: 56456.0000 - val_tp: 6.0000
    ## Epoch 21/30
    ## 112/112 - 0s - 2ms/step - fn: 41.0000 - fp: 8860.0000 - loss: 5.8042e-06 - precision: 0.0371 - recall: 0.8927 - tn: 218604.0000 - tp: 341.0000 - val_fn: 10.0000 - val_fp: 2791.0000 - val_loss: 0.2183 - val_precision: 0.0346 - val_recall: 0.9091 - val_tn: 54060.0000 - val_tp: 100.0000
    ## Epoch 22/30
    ## 112/112 - 0s - 2ms/step - fn: 15.0000 - fp: 9031.0000 - loss: 1.0670e-06 - precision: 0.0391 - recall: 0.9607 - tn: 218433.0000 - tp: 367.0000 - val_fn: 10.0000 - val_fp: 1275.0000 - val_loss: 0.0764 - val_precision: 0.0727 - val_recall: 0.9091 - val_tn: 55576.0000 - val_tp: 100.0000
    ## Epoch 23/30
    ## 112/112 - 0s - 2ms/step - fn: 10.0000 - fp: 5141.0000 - loss: 6.9193e-07 - precision: 0.0675 - recall: 0.9738 - tn: 222323.0000 - tp: 372.0000 - val_fn: 11.0000 - val_fp: 1872.0000 - val_loss: 0.0824 - val_precision: 0.0502 - val_recall: 0.9000 - val_tn: 54979.0000 - val_tp: 99.0000
    ## Epoch 24/30
    ## 112/112 - 0s - 2ms/step - fn: 2.0000 - fp: 4232.0000 - loss: 4.2135e-07 - precision: 0.0824 - recall: 0.9948 - tn: 223232.0000 - tp: 380.0000 - val_fn: 11.0000 - val_fp: 976.0000 - val_loss: 0.0488 - val_precision: 0.0921 - val_recall: 0.9000 - val_tn: 55875.0000 - val_tp: 99.0000
    ## Epoch 25/30
    ## 112/112 - 0s - 2ms/step - fn: 3.0000 - fp: 4149.0000 - loss: 3.7232e-07 - precision: 0.0837 - recall: 0.9921 - tn: 223315.0000 - tp: 379.0000 - val_fn: 13.0000 - val_fp: 359.0000 - val_loss: 0.0229 - val_precision: 0.2127 - val_recall: 0.8818 - val_tn: 56492.0000 - val_tp: 97.0000
    ## Epoch 26/30
    ## 112/112 - 0s - 2ms/step - fn: 3.0000 - fp: 3757.0000 - loss: 3.5368e-07 - precision: 0.0916 - recall: 0.9921 - tn: 223707.0000 - tp: 379.0000 - val_fn: 12.0000 - val_fp: 1111.0000 - val_loss: 0.0569 - val_precision: 0.0811 - val_recall: 0.8909 - val_tn: 55740.0000 - val_tp: 98.0000
    ## Epoch 27/30
    ## 112/112 - 0s - 2ms/step - fn: 2.0000 - fp: 3220.0000 - loss: 2.5512e-07 - precision: 0.1056 - recall: 0.9948 - tn: 224244.0000 - tp: 380.0000 - val_fn: 11.0000 - val_fp: 839.0000 - val_loss: 0.0396 - val_precision: 0.1055 - val_recall: 0.9000 - val_tn: 56012.0000 - val_tp: 99.0000
    ## Epoch 28/30
    ## 112/112 - 0s - 2ms/step - fn: 0.0000e+00 - fp: 2287.0000 - loss: 1.8121e-07 - precision: 0.1431 - recall: 1.0000 - tn: 225177.0000 - tp: 382.0000 - val_fn: 12.0000 - val_fp: 425.0000 - val_loss: 0.0272 - val_precision: 0.1874 - val_recall: 0.8909 - val_tn: 56426.0000 - val_tp: 98.0000
    ## Epoch 29/30
    ## 112/112 - 0s - 2ms/step - fn: 0.0000e+00 - fp: 1435.0000 - loss: 1.3034e-07 - precision: 0.2102 - recall: 1.0000 - tn: 226029.0000 - tp: 382.0000 - val_fn: 13.0000 - val_fp: 321.0000 - val_loss: 0.0250 - val_precision: 0.2321 - val_recall: 0.8818 - val_tn: 56530.0000 - val_tp: 97.0000
    ## Epoch 30/30
    ## 112/112 - 0s - 2ms/step - fn: 0.0000e+00 - fp: 1365.0000 - loss: 1.2132e-07 - precision: 0.2187 - recall: 1.0000 - tn: 226099.0000 - tp: 382.0000 - val_fn: 12.0000 - val_fp: 241.0000 - val_loss: 0.0183 - val_precision: 0.2891 - val_recall: 0.8909 - val_tn: 56610.0000 - val_tp: 98.0000

``` r
val_pred <- model %>%
  predict(val_features) %>%
  { as.integer(. > 0.5) }
```

    ## 1781/1781 - 1s - 427us/step

``` r
pred_correct <- val_df$Class == val_pred
cat(sprintf("Validation accuracy: %.2f", mean(pred_correct)))
```

    ## Validation accuracy: 1.00

``` r
fraudulent <- val_df$Class == 1

n_fraudulent_detected <- sum(fraudulent & pred_correct)
n_fraudulent_missed <- sum(fraudulent & !pred_correct)
n_legitimate_flagged <- sum(!fraudulent & !pred_correct)
```

## Conclusions

At the end of training, out of 56,961 validation transactions, we are:

- Correctly identifying 98 of them as fraudulent
- Missing 12 fraudulent transactions
- At the cost of incorrectly flagging 241 legitimate transactions

In the real world, one would put an even higher weight on class 1, so as
to reflect that False Negatives are more costly than False Positives.

Next time your credit card gets declined in an online purchase – this is
why.
