#' This script demonstrates how to use the Feature Spec interface implemented in
#' the tfdatasets package. We use the Porto Seguro dataset that can be downloaded
#' from Kaggle in this addres: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data

library(keras)
library(tfdatasets)
library(readr)
library(dplyr)

path <- "train.csv"

porto <- read_csv(path)

porto <- porto %>%
  mutate_at(vars(ends_with("cat")), as.character)

id_training <- sample.int(nrow(porto), size = 0.75*nrow(porto))
training <- porto[id_training,]
testing <- porto[-id_training,]

ft_spec <- training %>%
  select(-id) %>%
  feature_spec(target ~ .) %>%
  step_numeric_column(ends_with("bin")) %>%
  step_numeric_column(-ends_with("bin"), -ends_with("cat"), normalizer_fn = scaler_standard()) %>%
  step_categorical_column_with_vocabulary_list(ends_with("cat")) %>%
  step_embedding_column(ends_with("cat"), dimension = function(vocab_size) as.integer(sqrt(vocab_size) + 1)) %>%
  fit()

l <- 0.25

inputs <- layer_input_from_dataset(porto %>% select(-target))

output <- inputs %>%
  layer_dense_features(ft_spec$dense_features()) %>%
  layer_dense(units = 30, activation = "relu", kernel_regularizer = regularizer_l2(l)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 10, activation = "relu", kernel_regularizer = regularizer_l2(l)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 5, activation = "relu", kernel_regularizer = regularizer_l2(l)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 1, activation = "sigmoid", kernel_regularizer = regularizer_l2(l))

model <- keras_model(inputs, output)

auc <- tf$keras$metrics$AUC()

gini <- custom_metric(name = "gini", function(y_true, y_pred) {
  2*auc(y_true, y_pred) - 1
})

# Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003). Optimizing Classifier 
# Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
roc_auc_score <- function(y_true, y_pred) {
  
  pos = tf$boolean_mask(y_pred, tf$cast(y_true, tf$bool))
  neg = tf$boolean_mask(y_pred, !tf$cast(y_true, tf$bool))
  
  pos = tf$expand_dims(pos, 0L)
  neg = tf$expand_dims(neg, 1L)
  
  # original paper suggests performance is robust to exact parameter choice
  gamma = 0.2
  p     = 3
  
  difference = tf$zeros_like(pos * neg) + pos - neg - gamma
  
  masked = tf$boolean_mask(difference, difference < 0.0)
  
  tf$reduce_sum(tf$pow(-masked, p))
}

model %>%
  compile(
    loss = roc_auc_score,
    optimizer = optimizer_adam(),
    metrics = list(auc, gini)
  )

model %>%
  fit(
    x = training,
    y = training$target,
    epochs = 50,
    validation_data = list(testing, testing$target),
    batch_size = 512
  )

predictions <- predict(model, testing)
Metrics::auc(testing$target, predictions)

