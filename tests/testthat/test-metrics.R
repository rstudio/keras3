context("metrics")



test_succeeds("metrics can be used when compiling models", {
  define_model() %>%
    compile(
      loss='binary_crossentropy',
      optimizer = optimizer_sgd(),
      metrics=list(
        metric_binary_accuracy,
        metric_binary_crossentropy,
        metric_hinge
      )
    ) %>%
    fit(x = matrix(0, ncol = 784, nrow = 100), y = matrix(0, ncol = 10, nrow = 100),
        epochs = 1, verbose = 0)
})

test_succeeds("custom metrics can be used when compiling models", {

  metric_mean_pred <- custom_metric("mean_pred", function(y_true, y_pred) {
    k_mean(y_pred)
  })

  define_model() %>%
    compile(
      loss='binary_crossentropy',
      optimizer = optimizer_sgd(),
      metrics=list(
        metric_binary_accuracy,
        metric_binary_crossentropy,
        metric_hinge,
        metric_mean_pred
      )
    ) %>%
    fit(x = matrix(0, ncol = 784, nrow = 100), y = matrix(0, ncol = 10, nrow = 100),
        epochs = 1, verbose = 0)
})

test_succeeds("metrics be can called directly", {
  y_true <- k_constant(matrix(runif(100), nrow = 10, ncol = 10))
  y_pred <- k_constant(matrix(runif(100), nrow = 10, ncol = 10))
  metric_binary_accuracy(y_true, y_pred)
  metric_binary_crossentropy(y_true, y_pred)
  metric_hinge(y_true, y_pred)

  skip_if_cntk() # top_k doesn't work on CNTK, see
                 # https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-keras#known-issues)

  y_pred <- k_variable(matrix(c(0.3, 0.2, 0.1, 0.1, 0.2, 0.7), nrow=2, ncol = 3))
  y_true <- k_variable(matrix(c(0L, 1L), nrow = 2, ncol = 1))
  metric_top_k_categorical_accuracy(y_true, y_pred, k = 3)
  if (is_keras_available("2.0.5"))
    metric_sparse_top_k_categorical_accuracy(y_true, y_pred, k = 3)

})

test_succeeds("metrics for multiple output models", {

  input <- layer_input(shape = 1)

  output1 <- layer_dense(input, units = 1, name = "out1")
  output2 <- layer_dense(input, units = 1, name = "out2")

  model <- keras_model(input, list(output1, output2))

  model %>% compile(
    loss = "mse",
    optimizer = "adam",
    metrics = list(out1 = "mse", out2 = "mae")
  )

  history <- model %>% fit(
    x = matrix(0, ncol = 1, nrow = 100),
    y = list(rep(0, 100), rep(0, 100)),
    epochs = 1
  )

  if (tensorflow::tf_version() < "2.0") {
    expect_true(all(c("out2_mean_absolute_error", "out1_mean_squared_error") %in% names(history$metrics)))
    expect_true(all(!c("out1_mean_absolute_error", "out2_mean_squared_error") %in% names(history$metrics)))
  } else {
    expect_true(all(c("out2_mae", "out1_mse") %in% names(history$metrics)))
    expect_true(all(!c("out1_mae", "out2_mse") %in% names(history$metrics)))
  }

})


test_succeeds("get warning when passing using named list of metrics", {

  input <- layer_input(shape = 1)

  output1 <- layer_dense(input, units = 1, name = "out1")
  output2 <- layer_dense(input, units = 1, name = "out2")

  model <- keras_model(input, list(output1, output2))

  expect_warning({
    model %>% compile(
      loss = "mse",
      optimizer = "adam",
      metrics = list("metric1" = function(y_true, y_pred) k_mean(y_pred))
    )
  })

})


test_succeeds("get warning when passing Metric objects", {

   define_model() %>%
    compile(
      loss='binary_crossentropy',
      optimizer = optimizer_sgd(),
      metrics=list(
        metric_binary_accuracy(),
        metric_binary_crossentropy(),
        metric_hinge()
      )
    ) %>%
    fit(x = matrix(0, ncol = 784, nrow = 100), y = matrix(0, ncol = 10, nrow = 100),
        epochs = 1, verbose = 0)

})

N <- 100
X = random_array(c(N, 784))
Y = random_array(c(N, 10))
Y_sparse <- matrix(sample(0:9, N, TRUE))

test_metric <- function(metric, ...) {
  metric_name <- deparse(substitute(metric))
  loss <- "categorical_crossentropy"

  if(grepl("sparse", metric_name)) {
    Y <- Y_sparse
    loss <- "sparse_categorical_crossentropy"
  }

  test_that(metric_name, {
    m <- metric(...)

    expect_s3_class(m, c("keras.metrics.Metric",
                         'keras.metrics.base_metric.Metric'))

    define_model() %>%
      compile(loss = loss,
              optimizer = optimizer_sgd(),
              metrics = m) %>%
      fit(x = X, y = Y, epochs = 1, verbose = 0)
  })
}

test_metric(metric_sparse_categorical_accuracy)
test_metric(metric_sparse_categorical_crossentropy)
test_metric(metric_sparse_top_k_categorical_accuracy)

test_metric(metric_mean_squared_logarithmic_error)
test_metric(metric_binary_crossentropy)
test_metric(metric_precision_at_recall, recall = .5)
test_metric(metric_precision)
test_metric(metric_mean_absolute_percentage_error)
test_metric(metric_mean_absolute_error)
test_metric(metric_top_k_categorical_accuracy)
test_metric(metric_false_positives)
test_metric(metric_squared_hinge)
test_metric(metric_sensitivity_at_specificity, specificity= .5)
test_metric(metric_true_negatives)
test_metric(metric_recall)
test_metric(metric_hinge)
test_metric(metric_categorical_accuracy)
test_metric(metric_auc)
test_metric(metric_categorical_hinge)
test_metric(metric_binary_accuracy)
test_metric(metric_mean_squared_error)
test_metric(metric_specificity_at_sensitivity, sensitivity = .5)
test_metric(metric_accuracy)
test_metric(metric_false_negatives)
test_metric(metric_true_positives)
test_metric(metric_poisson)
test_metric(metric_logcosh_error)

test_metric(metric_root_mean_squared_error)
test_metric(metric_cosine_similarity)
test_metric(metric_mean_iou, num_classes = 10)
test_metric(metric_categorical_crossentropy)
test_metric(metric_kullback_leibler_divergence)

if(tf_version() >= "2.2")
  test_metric(metric_recall_at_precision, precision = .5)

if(tf_version() >= "2.6")
  test_metric(metric_mean_wrapper, fn = function(y_true, y_pred) {y_true})

## TODO: due to their unique signature, these don't work in the standard compile/fit API,
## only in standalone usage. Need to write custom tests for these.
# test_metric(metric_mean_tensor)
# test_metric(metric_sum)
# test_metric(metric_mean)
#
#' Example standalone usage:
#' m  <- metric_mean()
#' m$update_state(c(1, 3, 5, 7))
#' m$result()
#'
#' m$reset_state()
#' m$update_state(c(1, 3, 5, 7), sample_weight=c(1, 1, 0, 0))
#' m$result()
#' as.numeric(m$result())

## This metric seems to be affected by an upstream bug that prevents it from working in compile
## only works as a standalone metric presently
# test_metric(metric_mean_relative_error, normalizer = c(1, 3))

## deprecated
# test_metric(metric_cosine_proximity)


# asNamespace("keras") %>%
#   names() %>%
#   grep("^metric_", ., value = TRUE) %>%
#   sprintf("test_metric(%s)", .) %>%
#   cat(sep = "\n")
