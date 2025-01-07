

#' Computes the binary focal crossentropy loss.
#'
#' @description
#' According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002), it
#' helps to apply a focal factor to down-weight easy examples and focus more on
#' hard examples. By default, the focal tensor is computed as follows:
#'
#' `focal_factor = (1 - output)^gamma` for class 1
#' `focal_factor = output^gamma` for class 0
#' where `gamma` is a focusing parameter. When `gamma` = 0, there is no focal
#' effect on the binary crossentropy loss.
#'
#' If `apply_class_balancing == TRUE`, this function also takes into account a
#' weight balancing factor for the binary classes 0 and 1 as follows:
#'
#' `weight = alpha` for class 1 (`target == 1`)
#' `weight = 1 - alpha` for class 0
#' where `alpha` is a float in the range of `[0, 1]`.
#'
#' # Examples
#' ```{r}
#' y_true <- rbind(c(0, 1), c(0, 0))
#' y_pred <- rbind(c(0.6, 0.4), c(0.4, 0.6))
#' loss <- loss_binary_focal_crossentropy(y_true, y_pred, gamma=2)
#' loss
#' ```
#'
#' @returns
#' Binary focal crossentropy loss value
#' with shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param y_true
#' Ground truth values, of shape `(batch_size, d0, .. dN)`.
#'
#' @param y_pred
#' The predicted values, of shape `(batch_size, d0, .. dN)`.
#'
#' @param apply_class_balancing
#' A bool, whether to apply weight balancing on the
#' binary classes 0 and 1.
#'
#' @param alpha
#' A weight balancing factor for class 1, default is `0.25` as
#' mentioned in the reference. The weight for class 0 is `1.0 - alpha`.
#'
#' @param gamma
#' A focusing parameter, default is `2.0` as mentioned in the
#' reference.
#'
#' @param from_logits
#' Whether `y_pred` is expected to be a logits tensor. By
#' default, we assume that `y_pred` encodes a probability distribution.
#'
#' @param label_smoothing
#' Float in `[0, 1]`. If > `0` then smooth the labels by
#' squeezing them towards 0.5, that is,
#' using `1. - 0.5 * label_smoothing` for the target class
#' and `0.5 * label_smoothing` for the non-target class.
#'
#' @param axis
#' The axis along which the mean is computed. Defaults to `-1`.
#'
#' @inherit metric_binary_accuracy return
#' @export
#' @family losses
#' @family metrics
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/binary_focal_crossentropy>
#'
#' @tether keras.metrics.binary_focal_crossentropy
metric_binary_focal_crossentropy <-
function (y_true, y_pred, apply_class_balancing = FALSE, alpha = 0.25,
    gamma = 2, from_logits = FALSE, label_smoothing = 0, axis = -1L)
{
    args <- capture_args(list(axis = as_axis,
                              y_true = as_py_array,
                              y_pred = as_py_array))
    do.call(keras$metrics$binary_focal_crossentropy, args)
}


#' Computes the categorical focal crossentropy loss.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' y_true <- rbind(c(0, 1, 0), c(0, 0, 1))
#' y_pred <- rbind(c(0.05, 0.9, 0.05), c(0.1, 0.85, 0.05))
#' loss <- loss_categorical_focal_crossentropy(y_true, y_pred)
#' loss
#' ```
#'
#' @returns
#' Categorical focal crossentropy loss value.
#'
#' @param y_true
#' Tensor of one-hot true targets.
#'
#' @param y_pred
#' Tensor of predicted targets.
#'
#' @param alpha
#' A weight balancing factor for all classes, default is `0.25` as
#' mentioned in the reference. It can be a list of floats or a scalar.
#' In the multi-class case, alpha may be set by inverse class
#' frequency by using `compute_class_weight` from `sklearn.utils`.
#'
#' @param gamma
#' A focusing parameter, default is `2.0` as mentioned in the
#' reference. It helps to gradually reduce the importance given to
#' simple examples in a smooth manner. When `gamma` = 0, there is
#' no focal effect on the categorical crossentropy.
#'
#' @param from_logits
#' Whether `y_pred` is expected to be a logits tensor. By
#' default, we assume that `y_pred` encodes a probability
#' distribution.
#'
#' @param label_smoothing
#' Float in `[0, 1].` If > `0` then smooth the labels. For
#' example, if `0.1`, use `0.1 / num_classes` for non-target labels
#' and `0.9 + 0.1 / num_classes` for target labels.
#'
#' @param axis
#' Defaults to `-1`. The dimension along which the entropy is
#' computed.
#'
#' @inherit metric_binary_accuracy return
#' @export
#' @family losses
#' @family metrics
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/categorical_focal_crossentropy>
#'
#' @tether keras.metrics.categorical_focal_crossentropy
metric_categorical_focal_crossentropy <-
function (y_true, y_pred, alpha = 0.25, gamma = 2, from_logits = FALSE,
    label_smoothing = 0, axis = -1L)
{
  args <- capture_args(list(axis = as_axis,
                            y_true = as_py_array,
                            y_pred = as_py_array))
    do.call(keras$metrics$categorical_focal_crossentropy, args)
}


#' Computes Huber loss value.
#'
#' @description
#' Formula:
#' ```{r, eval = FALSE}
#' for (x in error) {
#'   if (abs(x) <= delta){
#'     loss <- c(loss, (0.5 * x^2))
#'   } else if (abs(x) > delta) {
#'     loss <- c(loss, (delta * abs(x) - 0.5 * delta^2))
#'   }
#' }
#' loss <- mean(loss)
#' ```
#' See: [Huber loss](https://en.wikipedia.org/wiki/Huber_loss).
#'
#' # Examples
#' ```{r}
#' y_true <- rbind(c(0, 1), c(0, 0))
#' y_pred <- rbind(c(0.6, 0.4), c(0.4, 0.6))
#' loss <- loss_huber(y_true, y_pred)
#' ```
#'
#' @returns
#' Tensor with one scalar loss entry per sample.
#'
#' @param y_true
#' tensor of true targets.
#'
#' @param y_pred
#' tensor of predicted targets.
#'
#' @param delta
#' A float, the point where the Huber loss function changes from a
#' quadratic to linear. Defaults to `1.0`.
#'
#' @inherit metric_binary_accuracy return
#' @export
#' @family losses
#' @family metrics
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/huber>
#'
#' @tether keras.metrics.huber
metric_huber <-
function (y_true, y_pred, delta = 1)
{
    args <- capture_args(list(y_true = as_py_array, y_pred = as_py_array))
    do.call(keras$metrics$huber, args)
}


#' Logarithm of the hyperbolic cosine of the prediction error.
#'
#' @description
#' Formula:
#' ```{r, eval = FALSE}
#' loss <- mean(log(cosh(y_pred - y_true)), axis=-1)
#' ```
#'
#' Note that `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small
#' `x` and to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works
#' mostly like the mean squared error, but will not be so strongly affected by
#' the occasional wildly incorrect prediction.
#'
#' # Examples
#' ```{r}
#' y_true <- rbind(c(0., 1.), c(0., 0.))
#' y_pred <- rbind(c(1., 1.), c(0., 0.))
#' loss <- metric_log_cosh(y_true, y_pred)
#' loss
#' ```
#'
#' @returns
#' Logcosh error values with shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param y_true
#' Ground truth values with shape = `[batch_size, d0, .. dN]`.
#'
#' @param y_pred
#' The predicted values with shape = `[batch_size, d0, .. dN]`.
#'
#' @inherit metric_binary_accuracy return
#' @export
#' @family losses
#' @family metrics
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/log_cosh>
#'
#' @tether keras.metrics.log_cosh
metric_log_cosh <-
function (y_true, y_pred)
{
    args <- capture_args(list(y_true = as_py_array, y_pred = as_py_array))
    do.call(keras$metrics$log_cosh, args)
}


#' Calculates how often predictions match binary labels.
#'
#' @description
#' This metric creates two local variables, `total` and `count` that are used
#' to compute the frequency with which `y_pred` matches `y_true`. This
#' frequency is ultimately returned as `binary accuracy`: an idempotent
#' operation that simply divides `total` by `count`.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_binary_accuracy()
#' m$update_state(rbind(1, 1, 0, 0), rbind(0.98, 1, 0, 0.6))
#' m$result()
#' # 0.75
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(rbind(1, 1, 0, 0), rbind(0.98, 1, 0, 0.6),
#'                sample_weight = c(1, 0, 0, 1))
#' m$result()
#' # 0.5
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(optimizer='sgd',
#'                   loss='binary_crossentropy',
#'                   metrics=list(metric_binary_accuracy()))
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param threshold
#' (Optional) Float representing the threshold for deciding
#' whether prediction values are 1 or 0.
#'
#' @param y_true
#' Tensor of true targets.
#'
#' @param y_pred
#' Tensor of predicted targets.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family accuracy metrics
#' @family metrics
#' @returns If `y_true` and `y_pred` are missing, a `Metric`
#'   instance is returned. The `Metric` instance that can be passed directly to
#'   `compile(metrics = )`, or used as a standalone object. See `?Metric` for
#'   example usage. If `y_true` and `y_pred` are provided, then a tensor with
#'   the computed value is returned.
#' @seealso
#' + <https://keras.io/api/metrics/accuracy_metrics#binaryaccuracy-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryAccuracy>
#'
#' @tether keras.metrics.BinaryAccuracy
metric_binary_accuracy <-
function (y_true, y_pred, threshold = 0.5, ..., name = "binary_accuracy",
    dtype = NULL)
{
    args <- capture_args(list(y_true = as_py_array, y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$BinaryAccuracy
    else keras$metrics$binary_accuracy
    do.call(callable, args)
}


#' Calculates how often predictions match one-hot labels.
#'
#' @description
#' You can provide logits of classes as `y_pred`, since argmax of
#' logits and probabilities are same.
#'
#' This metric creates two local variables, `total` and `count` that are used
#' to compute the frequency with which `y_pred` matches `y_true`. This
#' frequency is ultimately returned as `categorical accuracy`: an idempotent
#' operation that simply divides `total` by `count`.
#'
#' `y_pred` and `y_true` should be passed in as vectors of probabilities,
#' rather than as labels. If necessary, use `op_one_hot` to expand `y_true` as
#' a vector.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_categorical_accuracy()
#' m$update_state(rbind(c(0, 0, 1), c(0, 1, 0)), rbind(c(0.1, 0.9, 0.8),
#'                 c(0.05, 0.95, 0)))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(rbind(c(0, 0, 1), c(0, 1, 0)), rbind(c(0.1, 0.9, 0.8),
#'                c(0.05, 0.95, 0)),
#'                sample_weight = c(0.7, 0.3))
#' m$result()
#' # 0.3
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(optimizer = 'sgd',
#'                   loss = 'categorical_crossentropy',
#'                   metrics = list(metric_categorical_accuracy()))
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param y_true
#' Tensor of true targets.
#'
#' @param y_pred
#' Tensor of predicted targets.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family accuracy metrics
#' @family metrics
#' @inherit metric_binary_accuracy return
#' @seealso
#' + <https://keras.io/api/metrics/accuracy_metrics#categoricalaccuracy-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalAccuracy>
#'
#' @tether keras.metrics.CategoricalAccuracy
metric_categorical_accuracy <-
function (y_true, y_pred, ..., name = "categorical_accuracy",
    dtype = NULL)
{
    args <- capture_args(list(y_true = as_py_array, y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$CategoricalAccuracy
    else keras$metrics$categorical_accuracy
    do.call(callable, args)
}


#' Calculates how often predictions match integer labels.
#'
#' @description
#' ```{r, eval=FALSE}
#' acc <- sample_weight %*% (y_true == which.max(y_pred))
#' ```
#'
#' You can provide logits of classes as `y_pred`, since argmax of
#' logits and probabilities are same.
#'
#' This metric creates two local variables, `total` and `count` that are used
#' to compute the frequency with which `y_pred` matches `y_true`. This
#' frequency is ultimately returned as `sparse categorical accuracy`: an
#' idempotent operation that simply divides `total` by `count`.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_sparse_categorical_accuracy()
#' m$update_state(rbind(2L, 1L), rbind(c(0.1, 0.6, 0.3), c(0.05, 0.95, 0)))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(rbind(2L, 1L), rbind(c(0.1, 0.6, 0.3), c(0.05, 0.95, 0)),
#'                sample_weight = c(0.7, 0.3))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(optimizer = 'sgd',
#'                   loss = 'sparse_categorical_crossentropy',
#'                   metrics = list(metric_sparse_categorical_accuracy()))
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param y_true
#' Tensor of true targets.
#'
#' @param y_pred
#' Tensor of predicted targets.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family accuracy metrics
#' @family metrics
#' @inherit metric_binary_accuracy return
#' @seealso
#' + <https://keras.io/api/metrics/accuracy_metrics#sparsecategoricalaccuracy-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseCategoricalAccuracy>
#'
#' @tether keras.metrics.SparseCategoricalAccuracy
metric_sparse_categorical_accuracy <-
function (y_true, y_pred, ..., name = "sparse_categorical_accuracy",
    dtype = NULL)
{
    args <- capture_args(list(y_true = as_py_array,
                              y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$SparseCategoricalAccuracy
    else keras$metrics$sparse_categorical_accuracy
    do.call(callable, args)
}


#' Computes how often integer targets are in the top `K` predictions.
#'
#' @description
#'
#' Computes how often integer targets are in the top `K` predictions.
#'
#' By default, the arguments expected by `update_state()` are:
#' - `y_true`: a tensor of shape `(batch_size)` representing indices of true
#'     categories.
#' - `y_pred`: a tensor of shape `(batch_size, num_categories)` containing the
#'     scores for each sample for all possible categories.
#'
#' With `from_sorted_ids=TRUE`, the arguments expected by `update_state` are:
#' - `y_true`: a tensor of shape `(batch_size)` representing indices or IDs of
#'     true categories.
#' - `y_pred`: a tensor of shape `(batch_size, N)` containing the indices or
#'     IDs of the top `N` categories sorted in order from highest score to
#'     lowest score. `N` must be greater or equal to `k`.
#'
#' The `from_sorted_ids=TRUE` option can be more efficient when the set of
#' categories is very large and the model has an optimized way to retrieve the
#' top ones either without scoring or without maintaining the scores for all
#' the possible categories.
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_sparse_top_k_categorical_accuracy(k = 1L)
#' m$update_state(
#'   rbind(2, 1),
#'   op_array(rbind(c(0.1, 0.9, 0.8), c(0.05, 0.95, 0)), dtype = "float32")
#' )
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(
#'   rbind(2, 1),
#'   op_array(rbind(c(0.1, 0.9, 0.8), c(0.05, 0.95, 0)), dtype = "float32"),
#'   sample_weight = c(0.7, 0.3)
#' )
#' m$result()
#' ```
#'
#' ```{r}
#' m <- metric_sparse_top_k_categorical_accuracy(k = 1, from_sorted_ids = TRUE)
#' m$update_state(array(c(2, 1)), rbind(c(1, 0, 3),
#'                                      c(1, 2, 3)))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(optimizer = 'sgd',
#'                   loss = 'sparse_categorical_crossentropy',
#'                   metrics = list(metric_sparse_top_k_categorical_accuracy()))
#' ```
#'
#' @param k
#' (Optional) Number of top elements to look at for computing accuracy.
#' Defaults to `5`.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param from_sorted_ids
#' (Optional) When `FALSE`, the default, the tensor passed
#' in `y_pred` contains the unsorted scores of all possible categories.
#' When `TRUE`, `y_pred` contains the indices or IDs for the top
#' categories.
#'
#' @param y_true
#' Tensor of true targets.
#'
#' @param y_pred
#' Tensor of predicted targets.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_binary_accuracy return
#' @export
#' @family accuracy metrics
#' @family metrics
#' @seealso
#' + <https://keras.io/api/metrics/accuracy_metrics#sparsetopkcategoricalaccuracy-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseTopKCategoricalAccuracy>
#'
#' @tether keras.metrics.SparseTopKCategoricalAccuracy
metric_sparse_top_k_categorical_accuracy <-
function (y_true, y_pred, k = 5L, ..., name = "sparse_top_k_categorical_accuracy",
    dtype = NULL, from_sorted_ids = FALSE)
{
    args <- capture_args(list(k = as_integer,
                              y_true = as_py_array,
                              y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$SparseTopKCategoricalAccuracy
    else keras$metrics$sparse_top_k_categorical_accuracy
    do.call(callable, args)
}


#' Computes how often targets are in the top `K` predictions.
#'
#' @description
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_top_k_categorical_accuracy(k = 1)
#' m$update_state(
#'   rbind(c(0, 0, 1), c(0, 1, 0)),
#'   op_array(rbind(c(0.1, 0.9, 0.8), c(0.05, 0.95, 0)), dtype = "float32")
#' )
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(
#'   rbind(c(0, 0, 1), c(0, 1, 0)),
#'   op_array(rbind(c(0.1, 0.9, 0.8), c(0.05, 0.95, 0)), dtype = "float32"),
#'   sample_weight = c(0.7, 0.3))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model.compile(optimizer = 'sgd',
#'               loss = 'categorical_crossentropy',
#'               metrics = list(metric_top_k_categorical_accuracy()))
#' ```
#'
#' @param k
#' (Optional) Number of top elements to look at for computing accuracy.
#' Defaults to `5`.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param y_true
#' Tensor of true targets.
#'
#' @param y_pred
#' Tensor of predicted targets.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_binary_accuracy return
#' @export
#' @family accuracy metrics
#' @family metrics
#' @seealso
#' + <https://keras.io/api/metrics/accuracy_metrics#topkcategoricalaccuracy-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/TopKCategoricalAccuracy>
#'
#' @tether keras.metrics.TopKCategoricalAccuracy
metric_top_k_categorical_accuracy <-
function (y_true, y_pred, k = 5L, ..., name = "top_k_categorical_accuracy",
    dtype = NULL)
{
    args <- capture_args(list(k = as_integer,
                              y_true = as_py_array,
                              y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$TopKCategoricalAccuracy
    else keras$metrics$top_k_categorical_accuracy
    do.call(callable, args)
}


#' Approximates the AUC (Area under the curve) of the ROC or PR curves.
#'
#' @description
#' The AUC (Area under the curve) of the ROC (Receiver operating
#' characteristic; default) or PR (Precision Recall) curves are quality
#' measures of binary classifiers. Unlike the accuracy, and like cross-entropy
#' losses, ROC-AUC and PR-AUC evaluate all the operational points of a model.
#'
#' This class approximates AUCs using a Riemann sum. During the metric
#' accumulation phrase, predictions are accumulated within predefined buckets
#' by value. The AUC is then computed by interpolating per-bucket averages.
#' These buckets define the evaluated operational points.
#'
#' This metric creates four local variables, `true_positives`,
#' `true_negatives`, `false_positives` and `false_negatives` that are used to
#' compute the AUC.  To discretize the AUC curve, a linearly spaced set of
#' thresholds is used to compute pairs of recall and precision values. The area
#' under the ROC-curve is therefore computed using the height of the recall
#' values by the false positive rate, while the area under the PR-curve is the
#' computed using the height of the precision values by the recall.
#'
#' This value is ultimately returned as `auc`, an idempotent operation that
#' computes the area under a discretized curve of precision versus recall
#' values (computed using the aforementioned variables). The `num_thresholds`
#' variable controls the degree of discretization with larger numbers of
#' thresholds more closely approximating the true AUC. The quality of the
#' approximation may vary dramatically depending on `num_thresholds`. The
#' `thresholds` parameter can be used to manually specify thresholds which
#' split the predictions more evenly.
#'
#' For a best approximation of the real AUC, `predictions` should be
#' distributed approximately uniformly in the range `[0, 1]` (if
#' `from_logits=FALSE`). The quality of the AUC approximation may be poor if
#' this is not the case. Setting `summation_method` to 'minoring' or 'majoring'
#' can help quantify the error in the approximation by providing lower or upper
#' bound estimate of the AUC.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_auc(num_thresholds = 3)
#' m$update_state(c(0,   0,   1,   1),
#'                c(0, 0.5, 0.3, 0.9))
#' # threshold values are [0 - 1e-7, 0.5, 1 + 1e-7]
#' # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
#' # tp_rate = recall = [1, 0.5, 0], fp_rate = [1, 0, 0]
#' # auc = ((((1 + 0.5) / 2) * (1 - 0)) + (((0.5 + 0) / 2) * (0 - 0)))
#' #     = 0.75
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(c(0,   0,   1,   1),
#'                c(0, 0.5, 0.3, 0.9),
#'                sample_weight=c(1, 0, 0, 1))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' # Reports the AUC of a model outputting a probability.
#' model |> compile(
#'   optimizer = 'sgd',
#'   loss = loss_binary_crossentropy(),
#'   metrics = list(metric_auc())
#' )
#'
#' # Reports the AUC of a model outputting a logit.
#' model |> compile(
#'   optimizer = 'sgd',
#'   loss = loss_binary_crossentropy(from_logits = TRUE),
#'   metrics = list(metric_auc(from_logits = TRUE))
#' )
#' ```
#'
#' @param num_thresholds
#' (Optional) The number of thresholds to
#' use when discretizing the roc curve. Values must be > 1.
#' Defaults to `200`.
#'
#' @param curve
#' (Optional) Specifies the name of the curve to be computed,
#' `'ROC'` (default) or `'PR'` for the Precision-Recall-curve.
#'
#' @param summation_method
#' (Optional) Specifies the [Riemann summation method](
#' https://en.wikipedia.org/wiki/Riemann_sum) used.
#' 'interpolation' (default) applies mid-point summation scheme for
#' `ROC`.  For PR-AUC, interpolates (true/false) positives but not
#' the ratio that is precision (see Davis & Goadrich 2006 for
#' details); 'minoring' applies left summation for increasing
#' intervals and right summation for decreasing intervals; 'majoring'
#' does the opposite.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param thresholds
#' (Optional) A list of floating point values to use as the
#' thresholds for discretizing the curve. If set, the `num_thresholds`
#' parameter is ignored. Values should be in `[0, 1]`. Endpoint
#' thresholds equal to \{`-epsilon`, `1+epsilon`\} for a small positive
#' epsilon value will be automatically included with these to correctly
#' handle predictions equal to exactly 0 or 1.
#'
#' @param multi_label
#' boolean indicating whether multilabel data should be
#' treated as such, wherein AUC is computed separately for each label
#' and then averaged across labels, or (when `FALSE`) if the data
#' should be flattened into a single label before AUC computation. In
#' the latter case, when multilabel data is passed to AUC, each
#' label-prediction pair is treated as an individual data point. Should
#' be set to `FALSE`` for multi-class data.
#'
#' @param num_labels
#' (Optional) The number of labels, used when `multi_label` is
#' TRUE. If `num_labels` is not specified, then state variables get
#' created on the first call to `update_state`.
#'
#' @param label_weights
#' (Optional) list, array, or tensor of non-negative weights
#' used to compute AUCs for multilabel data. When `multi_label` is
#' TRUE, the weights are applied to the individual label AUCs when they
#' are averaged to produce the multi-label AUC. When it's FALSE, they
#' are used to weight the individual label predictions in computing the
#' confusion matrix on the flattened data. Note that this is unlike
#' `class_weights` in that `class_weights` weights the example
#' depending on the value of its label, whereas `label_weights` depends
#' only on the index of that label before flattening; therefore
#' `label_weights` should not be used for multi-class data.
#'
#' @param from_logits
#' boolean indicating whether the predictions (`y_pred` in
#' `update_state`) are probabilities or sigmoid logits. As a rule of thumb,
#' when using a keras loss, the `from_logits` constructor argument of the
#' loss should match the AUC `from_logits` constructor argument.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @returns a `Metric` instance is returned. The `Metric` instance can be passed
#'   directly to `compile(metrics = )`, or used as a standalone object. See
#'   `?Metric` for example usage.
#' @export
#' @family confusion metrics
#' @family metrics
#' @seealso
#' + <https://keras.io/api/metrics/classification_metrics#auc-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC>
#'
#' @tether keras.metrics.AUC
metric_auc <-
function (..., num_thresholds = 200L, curve = "ROC", summation_method = "interpolation",
    name = NULL, dtype = NULL, thresholds = NULL, multi_label = FALSE,
    num_labels = NULL, label_weights = NULL, from_logits = FALSE)
{
    args <- capture_args(list(num_thresholds = as_integer))
    do.call(keras$metrics$AUC, args)
}


#' Calculates the number of false negatives.
#'
#' @description
#' If `sample_weight` is given, calculates the sum of the weights of
#' false negatives. This metric creates one local variable, `accumulator`
#' that is used to keep track of the number of false negatives.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_false_negatives()
#' m$update_state(c(0, 1, 1, 1), c(0, 1, 0, 0))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(c(0, 1, 1, 1), c(0, 1, 0, 0), sample_weight=c(0, 0, 1, 0))
#' m$result()
#' # 1.0
#' ```
#'
#' @param thresholds
#' (Optional) Defaults to `0.5`. A float value, or a Python
#' list of float threshold values in `[0, 1]`. A threshold is
#' compared with prediction values to determine the truth value of
#' predictions (i.e., above the threshold is `TRUE`, below is `FALSE`).
#' If used with a loss function that sets `from_logits=TRUE` (i.e. no
#' sigmoid applied to predictions), `thresholds` should be set to 0.
#' One metric value is generated for each threshold value.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family confusion metrics
#' @family metrics
#' @seealso
#' + <https://keras.io/api/metrics/classification_metrics#falsenegatives-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/FALSENegatives>
#'
#' @tether keras.metrics.FalseNegatives
metric_false_negatives <-
function (..., thresholds = NULL, name = NULL, dtype = NULL)
{
    args <- capture_args()
    do.call(keras$metrics$FalseNegatives, args)
}


#' Calculates the number of false positives.
#'
#' @description
#' If `sample_weight` is given, calculates the sum of the weights of
#' false positives. This metric creates one local variable, `accumulator`
#' that is used to keep track of the number of false positives.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_false_positives()
#' m$update_state(c(0, 1, 0, 0), c(0, 0, 1, 1))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(c(0, 1, 0, 0), c(0, 0, 1, 1), sample_weight = c(0, 0, 1, 0))
#' m$result()
#' ```
#'
#' @param thresholds
#' (Optional) Defaults to `0.5`. A float value, or a Python
#' list of float threshold values in `[0, 1]`. A threshold is
#' compared with prediction values to determine the truth value of
#' predictions (i.e., above the threshold is `TRUE`, below is `FALSE`).
#' If used with a loss function that sets `from_logits=TRUE` (i.e. no
#' sigmoid applied to predictions), `thresholds` should be set to 0.
#' One metric value is generated for each threshold value.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family confusion metrics
#' @family metrics
#' @seealso
#' + <https://keras.io/api/metrics/classification_metrics#falsepositives-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/FALSEPositives>
#'
#' @tether keras.metrics.FalsePositives
metric_false_positives <-
function (..., thresholds = NULL, name = NULL, dtype = NULL)
{
    args <- capture_args()
    do.call(keras$metrics$FalsePositives, args)
}


#' Computes the precision of the predictions with respect to the labels.
#'
#' @description
#' The metric creates two local variables, `true_positives` and
#' `false_positives` that are used to compute the precision. This value is
#' ultimately returned as `precision`, an idempotent operation that simply
#' divides `true_positives` by the sum of `true_positives` and
#' `false_positives`.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' If `top_k` is set, we'll calculate precision as how often on average a class
#' among the top-k classes with the highest predicted values of a batch entry
#' is correct and can be found in the label for that entry.
#'
#' If `class_id` is specified, we calculate precision by considering only the
#' entries in the batch for which `class_id` is above the threshold and/or in
#' the top-k highest predictions, and computing the fraction of them for which
#' `class_id` is indeed a correct label.
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_precision()
#' m$update_state(c(0, 1, 1, 1),
#'                c(1, 0, 1, 1))
#' m$result() |> as.double() |> signif()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(c(0, 1, 1, 1),
#'                c(1, 0, 1, 1),
#'                sample_weight = c(0, 0, 1, 0))
#' m$result() |> as.double() |> signif()
#' ```
#'
#' ```{r}
#' # With top_k=2, it will calculate precision over y_true[1:2]
#' # and y_pred[1:2]
#' m <- metric_precision(top_k = 2)
#' m$update_state(c(0, 0, 1, 1), c(1, 1, 1, 1))
#' m$result()
#' ```
#'
#' ```{r}
#' # With top_k=4, it will calculate precision over y_true[1:4]
#' # and y_pred[1:4]
#' m <- metric_precision(top_k = 4)
#' m$update_state(c(0, 0, 1, 1), c(1, 1, 1, 1))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval=FALSE}
#' model |> compile(
#'   optimizer = 'sgd',
#'   loss = 'binary_crossentropy',
#'   metrics = list(metric_precision())
#' )
#' ```
#'
#' Usage with a loss with `from_logits=TRUE`:
#'
#' ```{r, eval = FALSE}
#' model |> compile(
#'   optimizer = 'adam',
#'   loss = loss_binary_crossentropy(from_logits = TRUE),
#'   metrics = list(metric_precision(thresholds = 0))
#' )
#' ```
#'
#' @param thresholds
#' (Optional) A float value, or a Python list of float
#' threshold values in `[0, 1]`. A threshold is compared with
#' prediction values to determine the truth value of predictions (i.e.,
#' above the threshold is `TRUE`, below is `FALSE`). If used with a
#' loss function that sets `from_logits=TRUE` (i.e. no sigmoid applied
#' to predictions), `thresholds` should be set to 0. One metric value
#' is generated for each threshold value. If neither `thresholds` nor
#' `top_k` are set, the default is to calculate precision with
#' `thresholds=0.5`.
#'
#' @param top_k
#' (Optional) Unset by default. An int value specifying the top-k
#' predictions to consider when calculating precision.
#'
#' @param class_id
#' (Optional) Integer class ID for which we want binary metrics.
#' This must be in the half-open interval `[0, num_classes)`, where
#' `num_classes` is the last dimension of predictions.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family confusion metrics
#' @family metrics
#' @seealso
#' + <https://keras.io/api/metrics/classification_metrics#precision-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Precision>
#'
#' @tether keras.metrics.Precision
metric_precision <-
function (..., thresholds = NULL, top_k = NULL, class_id = NULL,
    name = NULL, dtype = NULL)
{
    args <- capture_args(list(top_k = as_integer, class_id = as_integer))
    do.call(keras$metrics$Precision, args)
}


#' Computes best precision where recall is >= specified value.
#'
#' @description
#' This metric creates four local variables, `true_positives`,
#' `true_negatives`, `false_positives` and `false_negatives` that are used to
#' compute the precision at the given recall. The threshold for the given
#' recall value is computed and used to evaluate the corresponding precision.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' If `class_id` is specified, we calculate precision by considering only the
#' entries in the batch for which `class_id` is above the threshold
#' predictions, and computing the fraction of them for which `class_id` is
#' indeed a correct label.
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_precision_at_recall(recall = 0.5)
#' m$update_state(c(0,   0,   0,   1,   1),
#'                c(0, 0.3, 0.8, 0.3, 0.8))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(c(0,   0,   0,   1,   1),
#'                c(0, 0.3, 0.8, 0.3, 0.8),
#'                sample_weight = c(2, 2, 2, 1, 1))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model |> compile(
#'   optimizer = 'sgd',
#'   loss = 'binary_crossentropy',
#'   metrics = list(metric_precision_at_recall(recall = 0.8))
#' )
#' ```
#'
#' @param recall
#' A scalar value in range `[0, 1]`.
#'
#' @param num_thresholds
#' (Optional) Defaults to 200. The number of thresholds to
#' use for matching the given recall.
#'
#' @param class_id
#' (Optional) Integer class ID for which we want binary metrics.
#' This must be in the half-open interval `[0, num_classes)`, where
#' `num_classes` is the last dimension of predictions.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family confusion metrics
#' @family metrics
#' @seealso
#' + <https://keras.io/api/metrics/classification_metrics#precisionatrecall-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/PrecisionAtRecall>
#'
#' @tether keras.metrics.PrecisionAtRecall
metric_precision_at_recall <-
function (..., recall, num_thresholds = 200L, class_id = NULL,
    name = NULL, dtype = NULL)
{
    args <- capture_args(list(num_thresholds = as_integer, class_id = as_integer))
    do.call(keras$metrics$PrecisionAtRecall, args)
}


#' Computes the recall of the predictions with respect to the labels.
#'
#' @description
#' This metric creates two local variables, `true_positives` and
#' `false_negatives`, that are used to compute the recall. This value is
#' ultimately returned as `recall`, an idempotent operation that simply divides
#' `true_positives` by the sum of `true_positives` and `false_negatives`.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' If `top_k` is set, recall will be computed as how often on average a class
#' among the labels of a batch entry is in the top-k predictions.
#'
#' If `class_id` is specified, we calculate recall by considering only the
#' entries in the batch for which `class_id` is in the label, and computing the
#' fraction of them for which `class_id` is above the threshold and/or in the
#' top-k predictions.
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_recall()
#' m$update_state(c(0, 1, 1, 1),
#'                c(1, 0, 1, 1))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(c(0, 1, 1, 1),
#'                c(1, 0, 1, 1),
#'                sample_weight = c(0, 0, 1, 0))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model |> compile(
#'   optimizer = 'sgd',
#'   loss = 'binary_crossentropy',
#'   metrics = list(metric_recall())
#' )
#' ```
#'
#' Usage with a loss with `from_logits=TRUE`:
#'
#' ```{r, eval = FALSE}
#' model |> compile(
#'   optimizer = 'adam',
#'   loss = loss_binary_crossentropy(from_logits = TRUE),
#'   metrics = list(metric_recall(thresholds = 0))
#' )
#' ```
#'
#' @param thresholds
#' (Optional) A float value, or a Python list of float
#' threshold values in `[0, 1]`. A threshold is compared with
#' prediction values to determine the truth value of predictions (i.e.,
#' above the threshold is `TRUE`, below is `FALSE`). If used with a
#' loss function that sets `from_logits=TRUE` (i.e. no sigmoid
#' applied to predictions), `thresholds` should be set to 0.
#' One metric value is generated for each threshold value.
#' If neither `thresholds` nor `top_k` are set,
#' the default is to calculate recall with `thresholds=0.5`.
#'
#' @param top_k
#' (Optional) Unset by default. An int value specifying the top-k
#' predictions to consider when calculating recall.
#'
#' @param class_id
#' (Optional) Integer class ID for which we want binary metrics.
#' This must be in the half-open interval `[0, num_classes)`, where
#' `num_classes` is the last dimension of predictions.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family confusion metrics
#' @family metrics
#' @seealso
#' + <https://keras.io/api/metrics/classification_metrics#recall-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Recall>
#'
#' @tether keras.metrics.Recall
metric_recall <-
function (..., thresholds = NULL, top_k = NULL, class_id = NULL,
    name = NULL, dtype = NULL)
{
    args <- capture_args(list(top_k = as_integer, class_id = as_integer))
    do.call(keras$metrics$Recall, args)
}


#' Computes best recall where precision is >= specified value.
#'
#' @description
#' For a given score-label-distribution the required precision might not
#' be achievable, in this case 0.0 is returned as recall.
#'
#' This metric creates four local variables, `true_positives`,
#' `true_negatives`, `false_positives` and `false_negatives` that are used to
#' compute the recall at the given precision. The threshold for the given
#' precision value is computed and used to evaluate the corresponding recall.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' If `class_id` is specified, we calculate precision by considering only the
#' entries in the batch for which `class_id` is above the threshold
#' predictions, and computing the fraction of them for which `class_id` is
#' indeed a correct label.
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_recall_at_precision(precision = 0.8)
#' m$update_state(c(0,   0,   1,   1),
#'                c(0, 0.5, 0.3, 0.9))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(c(0,   0,   1,   1),
#'                c(0, 0.5, 0.3, 0.9),
#'                sample_weight = c(1, 0, 0, 1))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model |> compile(
#'   optimizer = 'sgd',
#'   loss = 'binary_crossentropy',
#'   metrics = list(metric_recall_at_precision(precision = 0.8))
#' )
#' ```
#'
#' @param precision
#' A scalar value in range `[0, 1]`.
#'
#' @param num_thresholds
#' (Optional) Defaults to 200. The number of thresholds
#' to use for matching the given precision.
#'
#' @param class_id
#' (Optional) Integer class ID for which we want binary metrics.
#' This must be in the half-open interval `[0, num_classes)`, where
#' `num_classes` is the last dimension of predictions.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family confusion metrics
#' @family metrics
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/RecallAtPrecision>
#'
#' @tether keras.metrics.RecallAtPrecision
metric_recall_at_precision <-
function (..., precision, num_thresholds = 200L, class_id = NULL,
    name = NULL, dtype = NULL)
{
    args <- capture_args(list(num_thresholds = as_integer, class_id = as_integer))
    do.call(keras$metrics$RecallAtPrecision, args)
}


#' Computes best sensitivity where specificity is >= specified value.
#'
#' @description
#' `Sensitivity` measures the proportion of actual positives that are correctly
#' identified as such `(tp / (tp + fn))`.
#' `Specificity` measures the proportion of actual negatives that are correctly
#' identified as such `(tn / (tn + fp))`.
#'
#' This metric creates four local variables, `true_positives`,
#' `true_negatives`, `false_positives` and `false_negatives` that are used to
#' compute the sensitivity at the given specificity. The threshold for the
#' given specificity value is computed and used to evaluate the corresponding
#' sensitivity.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' If `class_id` is specified, we calculate precision by considering only the
#' entries in the batch for which `class_id` is above the threshold
#' predictions, and computing the fraction of them for which `class_id` is
#' indeed a correct label.
#'
#' For additional information about specificity and sensitivity, see
#' [the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).
#'
#' # Usage
#'
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_sensitivity_at_specificity(specificity = 0.5)
#' m$update_state(c(0,   0,   0,   1,   1),
#'                c(0, 0.3, 0.8, 0.3, 0.8))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(c(0,   0,   0,   1,   1),
#'                c(0, 0.3, 0.8, 0.3, 0.8),
#'                sample_weight = c(1, 1, 2, 2, 1))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model |> compile(
#'   optimizer = 'sgd',
#'   loss = 'binary_crossentropy',
#'   metrics = list(metric_sensitivity_at_specificity())
#' )
#' ```
#'
#' @param specificity
#' A scalar value in range `[0, 1]`.
#'
#' @param num_thresholds
#' (Optional) Defaults to 200. The number of thresholds to
#' use for matching the given specificity.
#'
#' @param class_id
#' (Optional) Integer class ID for which we want binary metrics.
#' This must be in the half-open interval `[0, num_classes)`, where
#' `num_classes` is the last dimension of predictions.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family confusion metrics
#' @family metrics
#' @seealso
#' + <https://keras.io/api/metrics/classification_metrics#sensitivityatspecificity-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SensitivityAtSpecificity>
#'
#' @tether keras.metrics.SensitivityAtSpecificity
metric_sensitivity_at_specificity <-
function (..., specificity, num_thresholds = 200L, class_id = NULL,
    name = NULL, dtype = NULL)
{
    args <- capture_args(list(num_thresholds = as_integer, class_id = as_integer))
    do.call(keras$metrics$SensitivityAtSpecificity, args)
}


#' Computes best specificity where sensitivity is >= specified value.
#'
#' @description
#' `Sensitivity` measures the proportion of actual positives that are correctly
#' identified as such `(tp / (tp + fn))`.
#' `Specificity` measures the proportion of actual negatives that are correctly
#' identified as such `(tn / (tn + fp))`.
#'
#' This metric creates four local variables, `true_positives`,
#' `true_negatives`, `false_positives` and `false_negatives` that are used to
#' compute the specificity at the given sensitivity. The threshold for the
#' given sensitivity value is computed and used to evaluate the corresponding
#' specificity.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' If `class_id` is specified, we calculate precision by considering only the
#' entries in the batch for which `class_id` is above the threshold
#' predictions, and computing the fraction of them for which `class_id` is
#' indeed a correct label.
#'
#' For additional information about specificity and sensitivity, see
#' [the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_specificity_at_sensitivity(sensitivity = 0.5)
#' m$update_state(c(0,   0,   0,   1,   1),
#'                c(0, 0.3, 0.8, 0.3, 0.8))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(c(0,   0,   0,   1,   1),
#'                c(0, 0.3, 0.8, 0.3, 0.8),
#'                sample_weight = c(1, 1, 2, 2, 2))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model |> compile(
#'   optimizer = 'sgd',
#'   loss = 'binary_crossentropy',
#'   metrics = list(metric_sensitivity_at_specificity())
#' )
#' ```
#'
#' @param sensitivity
#' A scalar value in range `[0, 1]`.
#'
#' @param num_thresholds
#' (Optional) Defaults to 200. The number of thresholds to
#' use for matching the given sensitivity.
#'
#' @param class_id
#' (Optional) Integer class ID for which we want binary metrics.
#' This must be in the half-open interval `[0, num_classes)`, where
#' `num_classes` is the last dimension of predictions.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family confusion metrics
#' @family metrics
#' @seealso
#' + <https://keras.io/api/metrics/classification_metrics#specificityatsensitivity-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SpecificityAtSensitivity>
#'
#' @tether keras.metrics.SpecificityAtSensitivity
metric_specificity_at_sensitivity <-
function (..., sensitivity, num_thresholds = 200L, class_id = NULL,
    name = NULL, dtype = NULL)
{
    args <- capture_args(list(num_thresholds = as_integer, class_id = as_integer))
    do.call(keras$metrics$SpecificityAtSensitivity, args)
}


#' Calculates the number of true negatives.
#'
#' @description
#' If `sample_weight` is given, calculates the sum of the weights of
#' true negatives. This metric creates one local variable, `accumulator`
#' that is used to keep track of the number of true negatives.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_true_negatives()
#' m$update_state(c(0, 1, 0, 0), c(1, 1, 0, 0))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(c(0, 1, 0, 0), c(1, 1, 0, 0), sample_weight = c(0, 0, 1, 0))
#' m$result()
#' ```
#'
#' @param thresholds
#' (Optional) Defaults to `0.5`. A float value, or a Python
#' list of float threshold values in `[0, 1]`. A threshold is
#' compared with prediction values to determine the truth value of
#' predictions (i.e., above the threshold is `TRUE`, below is `FALSE`).
#' If used with a loss function that sets `from_logits=TRUE` (i.e. no
#' sigmoid applied to predictions), `thresholds` should be set to 0.
#' One metric value is generated for each threshold value.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family confusion metrics
#' @family metrics
#' @seealso
#' + <https://keras.io/api/metrics/classification_metrics#truenegatives-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/TrueNegatives>
#'
#' @tether keras.metrics.TrueNegatives
metric_true_negatives <-
function (..., thresholds = NULL, name = NULL, dtype = NULL)
{
    args <- capture_args()
    do.call(keras$metrics$TrueNegatives, args)
}


#' Calculates the number of true positives.
#'
#' @description
#' If `sample_weight` is given, calculates the sum of the weights of
#' true positives. This metric creates one local variable, `true_positives`
#' that is used to keep track of the number of true positives.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_true_positives()
#' m$update_state(c(0, 1, 1, 1), c(1, 0, 1, 1))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(c(0, 1, 1, 1), c(1, 0, 1, 1), sample_weight = c(0, 0, 1, 0))
#' m$result()
#' ```
#'
#' @param thresholds
#' (Optional) Defaults to `0.5`. A float value, or a Python
#' list of float threshold values in `[0, 1]`. A threshold is
#' compared with prediction values to determine the truth value of
#' predictions (i.e., above the threshold is `TRUE`, below is `FALSE`).
#' If used with a loss function that sets `from_logits=TRUE` (i.e. no
#' sigmoid applied to predictions), `thresholds` should be set to 0.
#' One metric value is generated for each threshold value.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family confusion metrics
#' @family metrics
#' @seealso
#' + <https://keras.io/api/metrics/classification_metrics#truepositives-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/TruePositives>
#'
#' @tether keras.metrics.TruePositives
metric_true_positives <-
function (..., thresholds = NULL, name = NULL, dtype = NULL)
{
    args <- capture_args()
    do.call(keras$metrics$TruePositives, args)
}


#' Computes F-1 Score.
#'
#' @description
#' Formula:
#'
#' ```{r, eval=FALSE}
#' f1_score <- 2 * (precision * recall) / (precision + recall)
#' ```
#' This is the harmonic mean of precision and recall.
#' Its output range is `[0, 1]`. It works for both multi-class
#' and multi-label classification.
#'
#' # Examples
#' ```{r}
#' metric <- metric_f1_score(threshold = 0.5)
#' y_true <- rbind(c(1, 1, 1),
#'                 c(1, 0, 0),
#'                 c(1, 1, 0))
#' y_pred <- rbind(c(0.2, 0.6, 0.7),
#'                 c(0.2, 0.6, 0.6),
#'                 c(0.6, 0.8, 0.0))
#' metric$update_state(y_true, y_pred)
#' result <- metric$result()
#' result
#' ```
#'
#' # Returns
#' F-1 Score: float.
#'
#' @param average
#' Type of averaging to be performed on data.
#' Acceptable values are `NULL`, `"micro"`, `"macro"`
#' and `"weighted"`. Defaults to `NULL`.
#' If `NULL`, no averaging is performed and `result()` will return
#' the score for each class.
#' If `"micro"`, compute metrics globally by counting the total
#' true positives, false negatives and false positives.
#' If `"macro"`, compute metrics for each label,
#' and return their unweighted mean.
#' This does not take label imbalance into account.
#' If `"weighted"`, compute metrics for each label,
#' and return their average weighted by support
#' (the number of true instances for each label).
#' This alters `"macro"` to account for label imbalance.
#' It can result in an score that is not between precision and recall.
#'
#' @param threshold
#' Elements of `y_pred` greater than `threshold` are
#' converted to be 1, and the rest 0. If `threshold` is
#' `NULL`, the argmax of `y_pred` is converted to 1, and the rest to 0.
#'
#' @param name
#' Optional. String name of the metric instance.
#'
#' @param dtype
#' Optional. Data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family f score metrics
#' @family metrics
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/F1Score>
#'
#' @tether keras.metrics.F1Score
metric_f1_score <-
function (..., average = NULL, threshold = NULL, name = "f1_score",
    dtype = NULL)
{
    args <- capture_args()
    do.call(keras$metrics$F1Score, args)
}


#' Computes F-Beta score.
#'
#' @description
#' Formula:
#'
#' ```{r, eval = FALSE}
#' b2 <- beta^2
#' f_beta_score <- (1 + b2) * (precision * recall) / (precision * b2 + recall)
#' ```
#' This is the weighted harmonic mean of precision and recall.
#' Its output range is `[0, 1]`. It works for both multi-class
#' and multi-label classification.
#'
#' # Examples
#' ```{r}
#' metric <- metric_fbeta_score(beta = 2.0, threshold = 0.5)
#' y_true <- rbind(c(1, 1, 1),
#'                 c(1, 0, 0),
#'                 c(1, 1, 0))
#' y_pred <- rbind(c(0.2, 0.6, 0.7),
#'                 c(0.2, 0.6, 0.6),
#'                 c(0.6, 0.8, 0.0))
#' metric$update_state(y_true, y_pred)
#' metric$result()
#' ```
#'
#' # Returns
#' F-Beta Score: float.
#'
#' @param average
#' Type of averaging to be performed across per-class results
#' in the multi-class case.
#' Acceptable values are `NULL`, `"micro"`, `"macro"` and
#' `"weighted"`. Defaults to `NULL`.
#' If `NULL`, no averaging is performed and `result()` will return
#' the score for each class.
#' If `"micro"`, compute metrics globally by counting the total
#' true positives, false negatives and false positives.
#' If `"macro"`, compute metrics for each label,
#' and return their unweighted mean.
#' This does not take label imbalance into account.
#' If `"weighted"`, compute metrics for each label,
#' and return their average weighted by support
#' (the number of true instances for each label).
#' This alters `"macro"` to account for label imbalance.
#' It can result in an score that is not between precision and recall.
#'
#' @param beta
#' Determines the weight of given to recall
#' in the harmonic mean between precision and recall (see pseudocode
#' equation above). Defaults to `1`.
#'
#' @param threshold
#' Elements of `y_pred` greater than `threshold` are
#' converted to be 1, and the rest 0. If `threshold` is
#' `NULL`, the argmax of `y_pred` is converted to 1, and the rest to 0.
#'
#' @param name
#' Optional. String name of the metric instance.
#'
#' @param dtype
#' Optional. Data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family f score metrics
#' @family metrics
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/FBetaScore>
#'
#' @tether keras.metrics.FBetaScore
metric_fbeta_score <-
function (..., average = NULL, beta = 1, threshold = NULL, name = "fbeta_score",
    dtype = NULL)
{
    args <- capture_args()
    do.call(keras$metrics$FBetaScore, args)
}


#' Computes the categorical hinge metric between `y_true` and `y_pred`.
#'
#' @description
#' Formula:
#'
#' ```{r, eval = FALSE}
#' loss <- maximum(neg - pos + 1, 0)
#' ```
#'
#' where `neg=maximum((1-y_true)*y_pred)` and `pos=sum(y_true*y_pred)`
#'
#' # Usage
#' Standalone usage:
#' ```{r}
#' m <- metric_categorical_hinge()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(0.6, 0.4), c(0.4, 0.6)))
#' m$result()
#'
#' m$reset_state()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(0.6, 0.4), c(0.4, 0.6)),
#'                sample_weight = c(1, 0))
#' m$result()
#' ```
#'
#' @returns
#' Categorical hinge loss values with shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param y_true
#' The ground truth values. `y_true` values are expected to be
#' either `{-1, +1}` or `{0, 1}` (i.e. a one-hot-encoded tensor) with
#' shape = `[batch_size, d0, .. dN]`.
#'
#' @param y_pred
#' The predicted values with shape = `[batch_size, d0, .. dN]`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_binary_accuracy return
#' @export
#' @family losses
#' @family metrics
#' @family hinge metrics
#' @seealso
#' + <https://keras.io/api/metrics/hinge_metrics#categoricalhinge-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalHinge>
#'
#' @tether keras.metrics.CategoricalHinge
metric_categorical_hinge <-
function (y_true, y_pred, ..., name = "categorical_hinge",
    dtype = NULL)
{
    args <- capture_args(list(y_true = as_py_array, y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$CategoricalHinge
    else keras$metrics$categorical_hinge
    do.call(callable, args)
}


#' Computes the hinge metric between `y_true` and `y_pred`.
#'
#' @description
#' Formula:
#'
#' ```{r, eval = FALSE}
#' loss <- mean(maximum(1 - y_true * y_pred, 0), axis=-1)
#' ```
#'
#' `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
#' provided we will convert them to -1 or 1.
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_hinge()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(0.6, 0.4), c(0.4, 0.6)))
#' m$result()
#'
#' m$reset_state()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(0.6, 0.4), c(0.4, 0.6)),
#'                sample_weight = c(1, 0))
#' m$result()
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param y_true
#' The ground truth values. `y_true` values are expected to be -1
#' or 1. If binary (0 or 1) labels are provided they will be converted
#' to -1 or 1 with shape = `[batch_size, d0, .. dN]`.
#'
#' @param y_pred
#' The predicted values with shape = `[batch_size, d0, .. dN]`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_binary_accuracy return
#' @export
#' @family losses
#' @family metrics
#' @family hinge metrics
#' @seealso
#' + <https://keras.io/api/metrics/hinge_metrics#hinge-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Hinge>
#'
#' @tether keras.metrics.Hinge
metric_hinge <-
function (y_true, y_pred, ..., name = "hinge", dtype = NULL)
{
    args <- capture_args(list(y_true = as_py_array, y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$Hinge
    else keras$metrics$hinge
    do.call(callable, args)
}


#' Computes the hinge metric between `y_true` and `y_pred`.
#'
#' @description
#' Formula:
#'
#' ```{r, eval = FALSE}
#' loss <- mean(square(maximum(1 - y_true * y_pred, 0)))
#' ```
#'
#' `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
#' provided we will convert them to -1 or 1.
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_squared_hinge()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(0.6, 0.4), c(0.4, 0.6)))
#' m$result()
#'
#' m$reset_state()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(0.6, 0.4), c(0.4, 0.6)),
#'                sample_weight = c(1, 0))
#' m$result()
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param y_true
#' The ground truth values. `y_true` values are expected to be -1
#' or 1. If binary (0 or 1) labels are provided we will convert them
#' to -1 or 1 with shape = `[batch_size, d0, .. dN]`.
#'
#' @param y_pred
#' The predicted values with shape = `[batch_size, d0, .. dN]`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_binary_accuracy return
#' @export
#' @family losses
#' @family metrics
#' @family hinge metrics
#' @seealso
#' + <https://keras.io/api/metrics/hinge_metrics#squaredhinge-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SquaredHinge>
#'
#' @tether keras.metrics.SquaredHinge
metric_squared_hinge <-
function (y_true, y_pred, ..., name = "squared_hinge",
    dtype = NULL)
{
    args <- capture_args(list(y_true = as_py_array, y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$SquaredHinge
    else keras$metrics$squared_hinge
    do.call(callable, args)
}


#' Computes the Intersection-Over-Union metric for class 0 and/or 1.
#'
#' @description
#' Formula:
#'
#' ```{r, eval = FALSE}
#' iou <- true_positives / (true_positives + false_positives + false_negatives)
#' ```
#' Intersection-Over-Union is a common evaluation metric for semantic image
#' segmentation.
#'
#' To compute IoUs, the predictions are accumulated in a confusion matrix,
#' weighted by `sample_weight` and the metric is then calculated from it.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' This class can be used to compute IoUs for a binary classification task
#' where the predictions are provided as logits. First a `threshold` is applied
#' to the predicted values such that those that are below the `threshold` are
#' converted to class 0 and those that are above the `threshold` are converted
#' to class 1.
#'
#' IoUs for classes 0 and 1 are then computed, the mean of IoUs for the classes
#' that are specified by `target_class_ids` is returned.
#'
#' # Note
#' with `threshold=0`, this metric has the same behavior as `IoU`.
#'
#' # Examples
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_binary_iou(target_class_ids=c(0L, 1L), threshold = 0.3)
#' m$update_state(c(0, 1, 0, 1), c(0.1, 0.2, 0.4, 0.7))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(c(0, 1, 0, 1), c(0.1, 0.2, 0.4, 0.7),
#'                sample_weight = c(0.2, 0.3, 0.4, 0.1))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(
#'     optimizer = 'sgd',
#'     loss = 'mse',
#'     metrics = list(metric_binary_iou(
#'         target_class_ids = 0L,
#'         threshold = 0.5
#'     ))
#' )
#' ```
#'
#' @param target_class_ids
#' A list or list of target class ids for which the
#' metric is returned. Options are `0`, `1`, or `c(0, 1)`. With
#' `0` (or `1`), the IoU metric for class 0 (or class 1,
#' respectively) is returned. With `c(0, 1)`, the mean of IoUs for the
#' two classes is returned.
#'
#' @param threshold
#' A threshold that applies to the prediction logits to convert
#' them to either predicted class 0 if the logit is below `threshold`
#' or predicted class 1 if the logit is above `threshold`.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family iou metrics
#' @family metrics
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryIoU>
#'
#' @tether keras.metrics.BinaryIoU
metric_binary_iou <-
function (..., target_class_ids = list(0L, 1L), threshold = 0.5,
    name = NULL, dtype = NULL)
{
    args <- capture_args()
    do.call(keras$metrics$BinaryIoU, args)
}


#' Computes the Intersection-Over-Union metric for specific target classes.
#'
#' @description
#' Formula:
#'
#' ```{r, eval=FALSE}
#' iou <- true_positives / (true_positives + false_positives + false_negatives)
#' ```
#' Intersection-Over-Union is a common evaluation metric for semantic image
#' segmentation.
#'
#' To compute IoUs, the predictions are accumulated in a confusion matrix,
#' weighted by `sample_weight` and the metric is then calculated from it.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' Note, this class first computes IoUs for all individual classes, then
#' returns the mean of IoUs for the classes that are specified by
#' `target_class_ids`. If `target_class_ids` has only one id value, the IoU of
#' that specific class is returned.
#'
#' # Examples
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_iou(num_classes = 2L, target_class_ids = list(0L))
#' m$update_state(c(0, 0, 1, 1), c(0, 1, 0, 1))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(c(0, 0, 1, 1), c(0, 1, 0, 1),
#'                sample_weight = c(0.3, 0.3, 0.3, 0.1))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval=FALSE}
#' model %>% compile(
#'   optimizer = 'sgd',
#'   loss = 'mse',
#'   metrics = list(metric_iou(num_classes = 2L, target_class_ids = list(0L))))
#' ```
#'
#' @param num_classes
#' The possible number of labels the prediction task can have.
#'
#' @param target_class_ids
#' A list of target class ids for which the
#' metric is returned. To compute IoU for a specific class, a list
#' of a single id value should be provided.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ignore_class
#' Optional integer. The ID of a class to be ignored during
#' metric computation. This is useful, for example, in segmentation
#' problems featuring a "void" class (commonly -1 or 255) in
#' segmentation maps. By default (`ignore_class=NULL`), all classes are
#'   considered.
#'
#' @param sparse_y_true
#' Whether labels are encoded using integers or
#' dense floating point vectors. If `FALSE`, the `argmax` function
#' is used to determine each sample's most likely associated label.
#'
#' @param sparse_y_pred
#' Whether predictions are encoded using integers or
#' dense floating point vectors. If `FALSE`, the `argmax` function
#' is used to determine each sample's most likely associated label.
#'
#' @param axis
#' (Optional) -1 is the dimension containing the logits.
#' Defaults to `-1`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family iou metrics
#' @family metrics
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/IoU>
#'
#' @tether keras.metrics.IoU
metric_iou <-
function (..., num_classes, target_class_ids, name = NULL, dtype = NULL,
    ignore_class = NULL, sparse_y_true = TRUE, sparse_y_pred = TRUE,
    axis = -1L)
{
    args <- capture_args(list(ignore_class = as_integer, axis = as_axis))
    do.call(keras$metrics$IoU, args)
}


#' Computes the mean Intersection-Over-Union metric.
#'
#' @description
#' Formula:
#'
#' ```{r, eval = FALSE}
#' iou <- true_positives / (true_positives + false_positives + false_negatives)
#' ```
#' Intersection-Over-Union is a common evaluation metric for semantic image
#' segmentation.
#'
#' To compute IoUs, the predictions are accumulated in a confusion matrix,
#' weighted by `sample_weight` and the metric is then calculated from it.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' Note that this class first computes IoUs for all individual classes, then
#' returns the mean of these values.
#'
#' # Examples
#' Standalone usage:
#'
#' ```{r}
#' # cm = [[1, 1],
#' #        [1, 1]]
#' # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
#' # iou = true_positives / (sum_row + sum_col - true_positives))
#' # result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2 = 0.33
#' m <- metric_mean_iou(num_classes = 2)
#' m$update_state(c(0, 0, 1, 1), c(0, 1, 0, 1))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(c(0, 0, 1, 1), c(0, 1, 0, 1),
#'                sample_weight=c(0.3, 0.3, 0.3, 0.1))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(
#'   optimizer = 'sgd',
#'   loss = 'mse',
#'   metrics = list(metric_mean_iou(num_classes=2)))
#' ```
#'
#' @param num_classes
#' The possible number of labels the prediction task can have.
#' This value must be provided, since a confusion matrix of dimension =
#' `[num_classes, num_classes]` will be allocated.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ignore_class
#' Optional integer. The ID of a class to be ignored during
#' metric computation. This is useful, for example, in segmentation
#' problems featuring a "void" class (commonly -1 or 255) in
#' segmentation maps. By default (`ignore_class=NULL`), all classes are
#' considered.
#'
#' @param sparse_y_true
#' Whether labels are encoded using integers or
#' dense floating point vectors. If `FALSE`, the `argmax` function
#' is used to determine each sample's most likely associated label.
#'
#' @param sparse_y_pred
#' Whether predictions are encoded using integers or
#' dense floating point vectors. If `FALSE`, the `argmax` function
#' is used to determine each sample's most likely associated label.
#'
#' @param axis
#' (Optional) The dimension containing the logits. Defaults to `-1`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family iou metrics
#' @family metrics
#' @seealso
#' + <https://keras.io/api/metrics/segmentation_metrics#meaniou-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU>
#'
#' @tether keras.metrics.MeanIoU
metric_mean_iou <-
function (..., num_classes, name = NULL, dtype = NULL, ignore_class = NULL,
    sparse_y_true = TRUE, sparse_y_pred = TRUE, axis = -1L)
{
    args <- capture_args(list(ignore_class = as_integer, axis = as_axis,
        num_classes = as_integer))
    do.call(keras$metrics$MeanIoU, args)
}


#' Computes the Intersection-Over-Union metric for one-hot encoded labels.
#'
#' @description
#' Formula:
#'
#' ```{r, eval = FALSE}
#' iou <- true_positives / (true_positives + false_positives + false_negatives)
#' ```
#' Intersection-Over-Union is a common evaluation metric for semantic image
#' segmentation.
#'
#' To compute IoUs, the predictions are accumulated in a confusion matrix,
#' weighted by `sample_weight` and the metric is then calculated from it.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' This class can be used to compute IoU for multi-class classification tasks
#' where the labels are one-hot encoded (the last axis should have one
#' dimension per class). Note that the predictions should also have the same
#' shape. To compute the IoU, first the labels and predictions are converted
#' back into integer format by taking the argmax over the class axis. Then the
#' same computation steps as for the base `IoU` class apply.
#'
#' Note, if there is only one channel in the labels and predictions, this class
#' is the same as class `IoU`. In this case, use `IoU` instead.
#'
#' Also, make sure that `num_classes` is equal to the number of classes in the
#' data, to avoid a "labels out of bound" error when the confusion matrix is
#' computed.
#'
#' # Examples
#' Standalone usage:
#'
#' ```{r}
#' y_true <- rbind(c(0, 0, 1),
#'                 c(1, 0, 0),
#'                 c(0, 1, 0),
#'                 c(1, 0, 0))
#' y_pred <- rbind(c(0.2, 0.3, 0.5),
#'                 c(0.1, 0.2, 0.7),
#'                 c(0.5, 0.3, 0.1),
#'                 c(0.1, 0.4, 0.5))
#' sample_weight <- c(0.1, 0.2, 0.3, 0.4)
#'
#' m <- metric_one_hot_iou(num_classes = 3, target_class_ids = c(0, 2))
#' m$update_state(y_true = y_true,
#'                y_pred = y_pred,
#'                sample_weight = sample_weight)
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(
#'   optimizer = 'sgd',
#'   loss = 'mse',
#'   metrics = list(metric_one_hot_iou(
#'     num_classes = 3L,
#'     target_class_id = list(1L)
#'   ))
#' )
#' ```
#'
#' @param num_classes
#' The possible number of labels the prediction task can have.
#'
#' @param target_class_ids
#' A list or list of target class ids for which the
#' metric is returned. To compute IoU for a specific class, a list
#' (or list) of a single id value should be provided.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ignore_class
#' Optional integer. The ID of a class to be ignored during
#' metric computation. This is useful, for example, in segmentation
#' problems featuring a "void" class (commonly -1 or 255) in
#' segmentation maps. By default (`ignore_class=NULL`), all classes are
#' considered.
#'
#' @param sparse_y_pred
#' Whether predictions are encoded using integers or
#' dense floating point vectors. If `FALSE`, the `argmax` function
#' is used to determine each sample's most likely associated label.
#'
#' @param axis
#' (Optional) The dimension containing the logits. Defaults to `-1`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family iou metrics
#' @family metrics
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/OneHotIoU>
#'
#' @tether keras.metrics.OneHotIoU
metric_one_hot_iou <-
function (..., num_classes, target_class_ids, name = NULL, dtype = NULL,
    ignore_class = NULL, sparse_y_pred = FALSE, axis = -1L)
{
    args <- capture_args(list(
      ignore_class = as_integer,
      axis = as_axis, num_classes = as_integer,
      target_class_ids = function (x) lapply(x, as_integer)))
    do.call(keras$metrics$OneHotIoU, args)
}


#' Computes mean Intersection-Over-Union metric for one-hot encoded labels.
#'
#' @description
#' Formula:
#'
#' ```{r, eval = FALSE}
#' iou <- true_positives / (true_positives + false_positives + false_negatives)
#' ```
#' Intersection-Over-Union is a common evaluation metric for semantic image
#' segmentation.
#'
#' To compute IoUs, the predictions are accumulated in a confusion matrix,
#' weighted by `sample_weight` and the metric is then calculated from it.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' This class can be used to compute the mean IoU for multi-class
#' classification tasks where the labels are one-hot encoded (the last axis
#' should have one dimension per class). Note that the predictions should also
#' have the same shape. To compute the mean IoU, first the labels and
#' predictions are converted back into integer format by taking the argmax over
#' the class axis. Then the same computation steps as for the base `MeanIoU`
#' class apply.
#'
#' Note, if there is only one channel in the labels and predictions, this class
#' is the same as class `metric_mean_iou`. In this case, use `metric_mean_iou` instead.
#'
#' Also, make sure that `num_classes` is equal to the number of classes in the
#' data, to avoid a "labels out of bound" error when the confusion matrix is
#' computed.
#'
#' # Examples
#' Standalone usage:
#'
#' ```{r}
#' y_true <- rbind(c(0, 0, 1), c(1, 0, 0), c(0, 1, 0), c(1, 0, 0))
#' y_pred <- rbind(c(0.2, 0.3, 0.5), c(0.1, 0.2, 0.7), c(0.5, 0.3, 0.1),
#'                 c(0.1, 0.4, 0.5))
#' sample_weight <- c(0.1, 0.2, 0.3, 0.4)
#' m <- metric_one_hot_mean_iou(num_classes = 3L)
#' m$update_state(
#'     y_true = y_true, y_pred = y_pred, sample_weight = sample_weight)
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(
#'     optimizer = 'sgd',
#'     loss = 'mse',
#'     metrics = list(metric_one_hot_mean_iou(num_classes = 3L)))
#' ```
#'
#' @param num_classes
#' The possible number of labels the prediction task can have.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ignore_class
#' Optional integer. The ID of a class to be ignored during
#' metric computation. This is useful, for example, in segmentation
#' problems featuring a "void" class (commonly -1 or 255) in
#' segmentation maps. By default (`ignore_class=NULL`), all classes are
#' considered.
#'
#' @param sparse_y_pred
#' Whether predictions are encoded using natural numbers or
#' probability distribution vectors. If `FALSE`, the `argmax`
#' function will be used to determine each sample's most likely
#' associated label.
#'
#' @param axis
#' (Optional) The dimension containing the logits. Defaults to `-1`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family iou metrics
#' @family metrics
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/OneHotMeanIoU>
#'
#' @tether keras.metrics.OneHotMeanIoU
metric_one_hot_mean_iou <-
function (..., num_classes, name = NULL, dtype = NULL, ignore_class = NULL,
    sparse_y_pred = FALSE, axis = -1L)
{
    args <- capture_args(list(ignore_class = as_integer, axis = as_axis,
        num_classes = as_integer))
    do.call(keras$metrics$OneHotMeanIoU, args)
}


#' Computes the crossentropy metric between the labels and predictions.
#'
#' @description
#' This is the crossentropy metric class to be used when there are only two
#' label classes (0 and 1).
#'
#' # Examples
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_binary_crossentropy()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(0.6, 0.4), c(0.4, 0.6)))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(0.6, 0.4), c(0.4, 0.6)),
#'                sample_weight=c(1, 0))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(
#'     optimizer = 'sgd',
#'     loss = 'mse',
#'     metrics = list(metric_binary_crossentropy()))
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param from_logits
#' (Optional) Whether output is expected
#' to be a logits tensor. By default, we consider
#' that output encodes a probability distribution.
#'
#' @param label_smoothing
#' (Optional) Float in `[0, 1]`.
#' When > 0, label values are smoothed,
#' meaning the confidence on label values are relaxed.
#' e.g. `label_smoothing=0.2` means that we will use
#' a value of 0.1 for label "0" and 0.9 for label "1".
#'
#' @param y_true
#' Ground truth values. shape = `[batch_size, d0, .. dN]`.
#'
#' @param y_pred
#' The predicted values. shape = `[batch_size, d0, .. dN]`.
#'
#' @param axis
#' The axis along which the mean is computed. Defaults to `-1`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_binary_accuracy return
#' @export
#' @family losses
#' @family metrics
#' @family probabilistic metrics
#' @seealso
#' + <https://keras.io/api/metrics/probabilistic_metrics#binarycrossentropy-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryCrossentropy>
#'
#' @tether keras.metrics.BinaryCrossentropy
metric_binary_crossentropy <-
function (y_true, y_pred, from_logits = FALSE, label_smoothing = 0,
    axis = -1L, ..., name = "binary_crossentropy", dtype = NULL)
{
    args <- capture_args(list(axis = as_axis,
                              y_true = as_py_array,
                              y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$BinaryCrossentropy
    else keras$metrics$binary_crossentropy
    do.call(callable, args)
}


#' Computes the crossentropy metric between the labels and predictions.
#'
#' @description
#' This is the crossentropy metric class to be used when there are multiple
#' label classes (2 or more). It assumes that labels are one-hot encoded,
#' e.g., when labels values are `c(2, 0, 1)`, then
#' `y_true` is `rbind(c([0, 0, 1), c(1, 0, 0), c(0, 1, 0))`.
#'
#' # Examples
#' Standalone usage:
#'
#' ```{r}
#' # EPSILON = 1e-7, y = y_true, y` = y_pred
#' # y` = clip_op_clip_by_value(output, EPSILON, 1. - EPSILON)
#' # y` = rbind(c(0.05, 0.95, EPSILON), c(0.1, 0.8, 0.1))
#' # xent = -sum(y * log(y'), axis = -1)
#' #      = -((log 0.95), (log 0.1))
#' #      = [0.051, 2.302]
#' # Reduced xent = (0.051 + 2.302) / 2
#'
#' m <- metric_categorical_crossentropy()
#' m$update_state(rbind(c(0, 1, 0), c(0, 0, 1)),
#'                rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1)))
#' m$result()
#' # 1.1769392
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(rbind(c(0, 1, 0), c(0, 0, 1)),
#'                rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1)),
#'                sample_weight = c(0.3, 0.7))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(
#'   optimizer = 'sgd',
#'   loss = 'mse',
#'   metrics = list(metric_categorical_crossentropy()))
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param from_logits
#' (Optional) Whether output is expected to be
#' a logits tensor. By default, we consider that output
#' encodes a probability distribution.
#'
#' @param label_smoothing
#' (Optional) Float in `[0, 1]`.
#' When > 0, label values are smoothed, meaning the confidence
#' on label values are relaxed. e.g. `label_smoothing=0.2` means
#' that we will use a value of 0.1 for label
#' "0" and 0.9 for label "1".
#'
#' @param axis
#' (Optional) Defaults to `-1`.
#' The dimension along which entropy is computed.
#'
#' @param y_true
#' Tensor of one-hot true targets.
#'
#' @param y_pred
#' Tensor of predicted targets.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_binary_accuracy return
#' @export
#' @family losses
#' @family metrics
#' @family probabilistic metrics
#' @seealso
#' + <https://keras.io/api/metrics/probabilistic_metrics#categoricalcrossentropy-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalCrossentropy>
#'
#' @tether keras.metrics.CategoricalCrossentropy
metric_categorical_crossentropy <-
function (y_true, y_pred, from_logits = FALSE, label_smoothing = 0,
    axis = -1L, ..., name = "categorical_crossentropy", dtype = NULL)
{
    args <- capture_args(list(axis = as_axis,
                              y_true = as_py_array,
                              y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$CategoricalCrossentropy
    else keras$metrics$categorical_crossentropy
    do.call(callable, args)
}


#' Computes Kullback-Leibler divergence metric between `y_true` and
#'
#' @description
#' Formula:
#'
#' ```{r, eval = FALSE}
#' loss <- y_true * log(y_true / y_pred)
#' ```
#'
#' `y_true` and `y_pred` are expected to be probability
#' distributions, with values between 0 and 1. They will get
#' clipped to the `[0, 1]` range.
#'
#' # Usage
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_kl_divergence()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(0.6, 0.4), c(0.4, 0.6)))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(0.6, 0.4), c(0.4, 0.6)),
#'                sample_weight = c(1, 0))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(optimizer = 'sgd',
#'                   loss = 'mse',
#'                   metrics = list(metric_kl_divergence()))
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param y_true
#' Tensor of true targets.
#'
#' @param y_pred
#' Tensor of predicted targets.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family losses
#' @family metrics
#' @family probabilistic metrics
#' @inherit metric_binary_accuracy return
#' @seealso
#' + <https://keras.io/api/metrics/probabilistic_metrics#kldivergence-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/KLDivergence>
#'
#' @tether keras.metrics.KLDivergence
metric_kl_divergence <-
function (y_true, y_pred, ..., name = "kl_divergence",
    dtype = NULL)
{
    args <- capture_args(list(y_true = as_py_array, y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$KLDivergence
    else keras$metrics$kl_divergence
    do.call(callable, args)
}


#' Computes the Poisson metric between `y_true` and `y_pred`.
#'
#' @description
#'
#' Formula:
#'
#' ```{r, eval = FALSE}
#' metric <- y_pred - y_true * log(y_pred)
#' ```
#'
#' # Examples
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_poisson()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)),
#'                sample_weight = c(1, 0))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(
#'   optimizer = 'sgd',
#'   loss = 'mse',
#'   metrics = list(metric_poisson())
#' )
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param y_true
#' Ground truth values. shape = `[batch_size, d0, .. dN]`.
#'
#' @param y_pred
#' The predicted values. shape = `[batch_size, d0, .. dN]`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family losses
#' @family metrics
#' @inherit metric_binary_accuracy return
#' @family probabilistic metrics
#' @seealso
#' + <https://keras.io/api/metrics/probabilistic_metrics#poisson-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Poisson>
#'
#' @tether keras.metrics.Poisson
metric_poisson <-
function (y_true, y_pred, ..., name = "poisson", dtype = NULL)
{
    args <- capture_args(list(y_true = as_py_array, y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$Poisson
    else keras$metrics$poisson
    do.call(callable, args)
}


#' Computes the crossentropy metric between the labels and predictions.
#'
#' @description
#' Use this crossentropy metric when there are two or more label classes.
#' It expects labels to be provided as integers. If you want to provide labels
#' that are one-hot encoded, please use the `metric_categorical_crossentropy()`
#' metric instead.
#'
#' There should be `num_classes` floating point values per feature for `y_pred`
#' and a single floating point value per feature for `y_true`.
#'
#' # Examples
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_sparse_categorical_crossentropy()
#' m$update_state(array(c(1, 2)),
#'                rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1)))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(array(c(1, 2)),
#'                rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1)),
#'                sample_weight = c(0.3, 0.7))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model |> compile(
#'   optimizer = 'sgd',
#'   loss = 'mse',
#'   metrics = list(metric_sparse_categorical_crossentropy())
#' )
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param from_logits
#' (Optional) Whether output is expected
#' to be a logits tensor. By default, we consider that output
#' encodes a probability distribution.
#'
#' @param axis
#' (Optional) Defaults to `-1`.
#' The dimension along which entropy is computed.
#'
#' @param y_true
#' Ground truth values.
#'
#' @param y_pred
#' The predicted values.
#'
#' @param ignore_class
#' Optional integer. The ID of a class to be ignored during
#' loss computation. This is useful, for example, in segmentation
#' problems featuring a "void" class (commonly -1 or 255) in
#' segmentation maps. By default (`ignore_class=NULL`), all classes are
#' considered.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family losses
#' @inherit metric_binary_accuracy return
#' @family metrics
#' @family probabilistic metrics
#' @seealso
#' + <https://keras.io/api/metrics/probabilistic_metrics#sparsecategoricalcrossentropy-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseCategoricalCrossentropy>
#'
#' @tether keras.metrics.SparseCategoricalCrossentropy
metric_sparse_categorical_crossentropy <-
function (y_true, y_pred, from_logits = FALSE, ignore_class = NULL,
    axis = -1L, ..., name = "sparse_categorical_crossentropy",
    dtype = NULL)
{
    args <- capture_args(list(axis = as_axis,
                              ignore_class = as_integer,
                              y_true = as_py_array,
                              y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$SparseCategoricalCrossentropy
    else keras$metrics$sparse_categorical_crossentropy
    do.call(callable, args)
}


#' Compute the (weighted) mean of the given values.
#'
#' @description
#' For example, if values is `c(1, 3, 5, 7)` then the mean is 4.
#' If `sample_weight` was specified as `c(1, 1, 0, 0)` then the mean would be 2.
#'
#' This metric creates two variables, `total` and `count`.
#' The mean value returned is simply `total` divided by `count`.
#'
#' # Examples
#' ```{r}
#' m <- metric_mean()
#' m$update_state(c(1, 3, 5, 7))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(c(1, 3, 5, 7), sample_weight = c(1, 1, 0, 0))
#' m$result()
#' ```
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family reduction metrics
#' @family metrics
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Mean>
#'
#' @tether keras.metrics.Mean
metric_mean <-
function (..., name = "mean", dtype = NULL)
{
    args <- capture_args()
    do.call(keras$metrics$Mean, args)
}


#' Wrap a stateless metric function with the `Mean` metric.
#'
#' @description
#' You could use this class to quickly build a mean metric from a function. The
#' function needs to have the signature `fn(y_true, y_pred)` and return a
#' per-sample loss array. `metric_mean_wrapper$result()` will return
#' the average metric value across all samples seen so far.
#'
#' For example:
#'
#' ```{r}
#' mse <- function(y_true, y_pred) {
#'   (y_true - y_pred)^2
#' }
#'
#' mse_metric <- metric_mean_wrapper(fn = mse)
#' mse_metric$update_state(c(0, 1), c(1, 1))
#' mse_metric$result()
#' ```
#'
#' @param fn
#' The metric function to wrap, with signature
#' `fn(y_true, y_pred)`.
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' Keyword arguments to pass on to `fn`.
#'
#' @inherit metric_auc return
#' @export
#' @family reduction metrics
#' @family metrics
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanMetricWrapper>
#'
#' @tether keras.metrics.MeanMetricWrapper
metric_mean_wrapper <-
function (..., fn, name = NULL, dtype = NULL)
{
    args <- capture_args(list(fn = function(x) as_py_function(
      x, default_name =
        if (is.null(name)) {
          if (is.symbol(fn_expr <- substitute(fn)))
            deparse(fn_expr)
          else
            "custom_metric"
        } else
        paste0(name, "_fn"))))
    do.call(keras$metrics$MeanMetricWrapper, args)
}


#' Compute the (weighted) sum of the given values.
#'
#' @description
#' For example, if `values` is `[1, 3, 5, 7]` then their sum is 16.
#' If `sample_weight` was specified as `[1, 1, 0, 0]` then the sum would be 4.
#'
#' This metric creates one variable, `total`.
#' This is ultimately returned as the sum value.
#'
#' # Examples
#' ```{r}
#' m <- metric_sum()
#' m$update_state(c(1, 3, 5, 7))
#' m$result()
#' ```
#'
#' ```{r}
#' m <- metric_sum()
#' m$update_state(c(1, 3, 5, 7), sample_weight = c(1, 1, 0, 0))
#' m$result()
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family reduction metrics
#' @family metrics
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Sum>
#'
#' @tether keras.metrics.Sum
metric_sum <-
function (..., name = "sum", dtype = NULL)
{
    args <- capture_args()
    do.call(keras$metrics$Sum, args)
}


#' Computes the cosine similarity between the labels and predictions.
#'
#' @description
#' Formula:
#'
#' ```{r, eval=FALSE}
#' loss <- sum(l2_norm(y_true) * l2_norm(y_pred))
#' ```
#' See: [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
#' This metric keeps the average cosine similarity between `predictions` and
#' `labels` over a stream of data.
#'
#' # Examples
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_cosine_similarity(axis=2)
#' m$update_state(rbind(c(0., 1.), c(1., 1.)), rbind(c(1., 0.), c(1., 1.)))
#' m$result()
#'
#' m$reset_state()
#' m$update_state(rbind(c(0., 1.), c(1., 1.)), rbind(c(1., 0.), c(1., 1.)),
#'                sample_weight = c(0.3, 0.7))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(
#'   optimizer = 'sgd',
#'   loss = 'mse',
#'   metrics = list(metric_cosine_similarity(axis=2)))
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param axis
#' (Optional) Defaults to `-1`. The dimension along which the cosine
#' similarity is computed.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family regression metrics
#' @family metrics
#' @seealso
#' + <https://keras.io/api/metrics/regression_metrics#cosinesimilarity-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CosineSimilarity>
#'
#' @tether keras.metrics.CosineSimilarity
metric_cosine_similarity <-
function (..., name = "cosine_similarity", dtype = NULL, axis = -1L)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$metrics$CosineSimilarity, args)
}


#' Computes the logarithm of the hyperbolic cosine of the prediction error.
#'
#' @description
#' Formula:
#'
#' ```{r, eval = FALSE}
#' error <- y_pred - y_true
#' logcosh <- mean(log((exp(error) + exp(-error))/2), axis=-1)
#' ```
#'
#' # Examples
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_log_cosh_error()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)))
#' m$result()
#'
#' m$reset_state()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)),
#'                sample_weight = c(1, 0))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(optimizer = 'sgd',
#'                   loss = 'mse',
#'                   metrics = list(metric_log_cosh_error()))
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family regression metrics
#' @family metrics
#' @seealso
#' + <https://keras.io/api/metrics/regression_metrics#logcosherror-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/LogCoshError>
#'
#' @tether keras.metrics.LogCoshError
metric_log_cosh_error <-
function (..., name = "logcosh", dtype = NULL)
{
    args <- capture_args()
    do.call(keras$metrics$LogCoshError, args)
}


#' Computes the mean absolute error between the labels and predictions.
#'
#' @description
#'
#' Formula:
#'
#' ```{r, eval = FALSE}
#' loss <- mean(abs(y_true - y_pred))
#' ```
#'
#' # Examples
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_mean_absolute_error()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)))
#' m$result()
#'
#' m$reset_state()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)),
#'                sample_weight = c(1, 0))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(
#'     optimizer = 'sgd',
#'     loss = 'mse',
#'     metrics = list(metric_mean_absolute_error()))
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param y_true
#' Ground truth values with shape = `[batch_size, d0, .. dN]`.
#'
#' @param y_pred
#' The predicted values with shape = `[batch_size, d0, .. dN]`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @inherit metric_binary_accuracy return
#' @family losses
#' @family metrics
#' @family regression metrics
#' @seealso
#' + <https://keras.io/api/metrics/regression_metrics#meanabsoluteerror-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanAbsoluteError>
#'
#' @tether keras.metrics.MeanAbsoluteError
metric_mean_absolute_error <-
function (y_true, y_pred, ..., name = "mean_absolute_error",
    dtype = NULL)
{
    args <- capture_args(list(y_true = as_py_array, y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$MeanAbsoluteError
    else keras$metrics$mean_absolute_error
    do.call(callable, args)
}


#' Computes mean absolute percentage error between `y_true` and `y_pred`.
#'
#' @description
#' Formula:
#'
#' ```{r, eval = FALSE}
#' loss <- 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)
#' ```
#'
#' Division by zero is prevented by dividing by `maximum(y_true, epsilon)`
#' where `epsilon = keras$backend$epsilon()`
#' (default to `1e-7`).
#'
#' # Examples
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_mean_absolute_percentage_error()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)))
#' m$result()
#'
#' m$reset_state()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)),
#'                sample_weight = c(1, 0))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(
#'     optimizer = 'sgd',
#'     loss = 'mse',
#'     metrics = list(metric_mean_absolute_percentage_error()))
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param y_true
#' Ground truth values with shape = `[batch_size, d0, .. dN]`.
#'
#' @param y_pred
#' The predicted values with shape = `[batch_size, d0, .. dN]`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_binary_accuracy return
#' @export
#' @family losses
#' @family metrics
#' @family regression metrics
#' @seealso
#' + <https://keras.io/api/metrics/regression_metrics#meanabsolutepercentageerror-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanAbsolutePercentageError>
#'
#' @tether keras.metrics.MeanAbsolutePercentageError
metric_mean_absolute_percentage_error <-
function (y_true, y_pred, ..., name = "mean_absolute_percentage_error",
    dtype = NULL)
{
    args <- capture_args(list(y_true = as_py_array,
                              y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$MeanAbsolutePercentageError
    else keras$metrics$mean_absolute_percentage_error
    do.call(callable, args)
}


#' Computes the mean squared error between `y_true` and `y_pred`.
#'
#' @description
#'
#' Formula:
#'
#' ```{r, eval = FALSE}
#' loss <- mean(square(y_true - y_pred))
#' ```
#'
#' # Examples
#' ```{r}
#' m <- metric_mean_squared_error()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)))
#' m$result()
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param y_true
#' Ground truth values with shape = `[batch_size, d0, .. dN]`.
#'
#' @param y_pred
#' The predicted values with shape = `[batch_size, d0, .. dN]`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_binary_accuracy return
#' @export
#' @family losses
#' @family metrics
#' @family regression metrics
#' @seealso
#' + <https://keras.io/api/metrics/regression_metrics#meansquarederror-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanSquaredError>
#'
#' @tether keras.metrics.MeanSquaredError
metric_mean_squared_error <-
function (y_true, y_pred, ..., name = "mean_squared_error",
    dtype = NULL)
{
    args <- capture_args(list(y_true = as_py_array,
                              y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$MeanSquaredError
    else keras$metrics$mean_squared_error
    do.call(callable, args)
}


#' Computes mean squared logarithmic error between `y_true` and `y_pred`.
#'
#' @description
#' Formula:
#'
#' ```{r, eval = FALSE}
#' loss <- mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)
#' ```
#'
#' Note that `y_pred` and `y_true` cannot be less or equal to 0. Negative
#' values and 0 values will be replaced with `keras$backend$epsilon()`
#' (default to `1e-7`).
#'
#' # Examples
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_mean_squared_logarithmic_error()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)))
#' m$result()
#'
#' m$reset_state()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)),
#'                sample_weight = c(1, 0))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(
#'   optimizer = 'sgd',
#'   loss = 'mse',
#'   metrics = list(metric_mean_squared_logarithmic_error()))
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param y_true
#' Ground truth values with shape = `[batch_size, d0, .. dN]`.
#'
#' @param y_pred
#' The predicted values with shape = `[batch_size, d0, .. dN]`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_binary_accuracy return
#' @export
#' @family losses
#' @family metrics
#' @family regression metrics
#' @seealso
#' + <https://keras.io/api/metrics/regression_metrics#meansquaredlogarithmicerror-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanSquaredLogarithmicError>
#'
#' @tether keras.metrics.MeanSquaredLogarithmicError
metric_mean_squared_logarithmic_error <-
function (y_true, y_pred, ..., name = "mean_squared_logarithmic_error",
    dtype = NULL)
{
    args <- capture_args(list(y_true = as_py_array,
                              y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$MeanSquaredLogarithmicError
    else keras$metrics$mean_squared_logarithmic_error
    do.call(callable, args)
}


#' Computes R2 score.
#'
#' @description
#' Formula:
#'
#' ```{r, eval = FALSE}
#' sum_squares_residuals <- sum((y_true - y_pred) ** 2)
#' sum_squares <- sum((y_true - mean(y_true)) ** 2)
#' R2 <- 1 - sum_squares_residuals / sum_squares
#' ```
#'
#' This is also called the
#' [coefficient of determination](
#' https://en.wikipedia.org/wiki/Coefficient_of_determination).
#'
#' It indicates how close the fitted regression line
#' is to ground-truth data.
#'
#' - The highest score possible is 1.0. It indicates that the predictors
#'     perfectly accounts for variation in the target.
#' - A score of 0.0 indicates that the predictors do not
#'     account for variation in the target.
#' - It can also be negative if the model is worse than random.
#'
#' This metric can also compute the "Adjusted R2" score.
#'
#' # Examples
#' ```{r}
#' y_true <- rbind(1, 4, 3)
#' y_pred <- rbind(2, 4, 4)
#' metric <- metric_r2_score()
#' metric$update_state(y_true, y_pred)
#' metric$result()
#' ```
#'
#' @param class_aggregation
#' Specifies how to aggregate scores corresponding to
#' different output classes (or target dimensions),
#' i.e. different dimensions on the last axis of the predictions.
#' Equivalent to `multioutput` argument in Scikit-Learn.
#' Should be one of
#' `NULL` (no aggregation), `"uniform_average"`,
#' `"variance_weighted_average"`.
#'
#' @param num_regressors
#' Number of independent regressors used
#' ("Adjusted R2" score). 0 is the standard R2 score.
#' Defaults to `0`.
#'
#' @param name
#' Optional. string name of the metric instance.
#'
#' @param dtype
#' Optional. data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family regression metrics
#' @family metrics
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/R2Score>
#'
#' @tether keras.metrics.R2Score
metric_r2_score <-
function (..., class_aggregation = "uniform_average", num_regressors = 0L,
    name = "r2_score", dtype = NULL)
{
    args <- capture_args(list(num_regressors = as_integer))
    do.call(keras$metrics$R2Score, args)
}


#' Computes root mean squared error metric between `y_true` and `y_pred`.
#'
#' @description
#' Formula:
#'
#' ```{r, eval = FALSE}
#' loss <- sqrt(mean((y_pred - y_true) ^ 2))
#' ```
#'
#' # Examples
#' Standalone usage:
#'
#' ```{r}
#' m <- metric_root_mean_squared_error()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)))
#' m$result()
#' ```
#'
#' ```{r}
#' m$reset_state()
#' m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)),
#'                sample_weight = c(1, 0))
#' m$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(
#'   optimizer = 'sgd',
#'   loss = 'mse',
#'   metrics = list(metric_root_mean_squared_error()))
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit metric_auc return
#' @export
#' @family regression metrics
#' @family metrics
#' @seealso
#' + <https://keras.io/api/metrics/regression_metrics#rootmeansquarederror-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/RootMeanSquaredError>
#'
#' @tether keras.metrics.RootMeanSquaredError
metric_root_mean_squared_error <-
function (..., name = "root_mean_squared_error", dtype = NULL)
{
    args <- capture_args()
    do.call(keras$metrics$RootMeanSquaredError, args)
}


#' Calculates the Concordance Correlation Coefficient (CCC).
#'
#' @description
#' Formula:
#'
#' ```r
#' loss <- mean(
#'   2 * (y_true - mean(y_true)) * (y_pred - mean(y_pred)) /
#'     (var(y_true) + var(y_pred) + (mean(y_true) - mean(y_pred))^2)
#' )
#' ```
#'
#' CCC evaluates the agreement between true values (`y_true`) and predicted
#' values (`y_pred`) by considering both precision and accuracy. The
#' coefficient ranges from -1 to 1, where a value of 1 indicates perfect
#' agreement.
#'
#' This metric is useful in regression tasks where it is important to assess
#' how well the predictions match the true values, taking into account both
#' their correlation and proximity to the 45-degree line of perfect
#' concordance.
#'
#' # Examples
#' ```{r}
#' ccc <- metric_concordance_correlation(axis=-1)
#' y_true <- rbind(c(0, 1, 0.5),
#'                 c(1, 1, 0.2))
#' y_pred <- rbind(c(0.1, 0.9, 0.5),
#'                 c(1, 0.9, 0.2))
#' ccc$update_state(y_true, y_pred)
#' ccc$result()
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```r
#' model |> compile(
#'   optimizer = 'sgd',
#'   loss = 'mean_squared_error',
#'   metrics = c(metric_concordance_correlation())
#' )
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param axis
#' (Optional) integer or tuple of integers of the axis/axes along
#' which to compute the metric. Defaults to `-1`.
#'
#' @param y_true
#' Tensor of true targets.
#'
#' @param y_pred
#' Tensor of predicted targets.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family regression metrics
#' @family metrics
#' @export
#' @tether keras.metrics.ConcordanceCorrelation
metric_concordance_correlation <-
function (y_true, y_pred, axis = -1L, ...,
          name = "concordance_correlation", dtype = NULL)
{
  args <- capture_args(list(
    y_true = as_py_array,
    y_pred = as_py_array,
    axis = as_axis
  ))
  callable <- if (missing(y_true) && missing(y_pred))
    keras$metrics$ConcordanceCorrelation
  else
    keras$metrics$concordance_correlation
  do.call(callable, args)
}


#' @importFrom reticulate py_to_r_wrapper
#' @export
#' @keywords internal
py_to_r_wrapper.keras.src.metrics.metric.Metric <- py_to_r_wrapper.keras.src.losses.loss.Loss


# --------------------------------------------------------------------------------



# .metric_return_roxygen <- function(has_function_handle = FALSE) {
#   if(has_function_handle) {
# r"---(
# @returns
# If `y_true` and `y_pred` are missing, a (subclassed) `Metric`
# instance is returned. The `Metric` object can be passed directly to
# `compile(metrics = )` or used as a standalone object. See `?`[`Metric`] for
# example usage.
#
# Alternatively, if called with `y_true` and `y_pred` arguments, then the
# computed case-wise values for the mini-batch are returned directly.
# )---"
#   } else {
# r"---(
# @returns
# A (subclassed) `Metric` instance that can be passed directly to
# `compile(metrics = )`, or used as a standalone object. See `?`[`Metric`] for
# example usage.
# )---"
#   }
# }


#' Custom metric function
#'
#' @param name name used to show training progress output
#' @param metric_fn An R function with signature `function(y_true, y_pred)`
#'   that accepts tensors.
#'
#' @details
#' You can provide an arbitrary R function as a custom metric. Note that
#' the `y_true` and `y_pred` parameters are tensors, so computations on
#' them should use `op_*` tensor functions.
#'
#' Use the `custom_metric()` function to define a custom metric.
#' Note that a name (`'mean_pred'`) is provided for the custom metric
#' function: this name is used within training progress output.
#'
#' If you want to save and load a model with custom metrics, you should
#' also call [`register_keras_serializable()`], or
#' specify the metric in the call the [load_model()]. For example:
#' `load_model("my_model.keras", c('mean_pred' = metric_mean_pred))`.
#'
#' Alternatively, you can wrap all of your code in a call to
#' [with_custom_object_scope()] which will allow you to refer to the
#' metric by name just like you do with built in keras metrics.
#'
#'
#' Alternative ways of supplying custom metrics:
#'  +  `custom_metric():` Arbitrary R function.
#'  +  [metric_mean_wrapper()]: Wrap an arbitrary R function in a `Metric` instance.
#'  +  Create a custom [`Metric()`] subclass.
#'
#' @returns A callable function with a `__name__` attribute.
#' @family metrics
#' @export
custom_metric <- function(name, metric_fn) {
  py_func2(metric_fn, convert = TRUE, name = name)
}
