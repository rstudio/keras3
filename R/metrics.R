
#' Metric
#'
#' A `Metric` object encapsulates metric logic and state that can be used to
#' track model performance during training. It is what is returned by the family
#' of metric functions that start with prefix `metric_*`.
#'
#' @param name (Optional) string name of the metric instance.
#' @param dtype (Optional) data type of the metric result.
#'
#' @returns A (subclassed) `Metric` instance that can be passed directly to
#'   `compile(metrics = )`, or used as a standalone object. See `?Metric` for
#'   example usage.
#'
#'
#' @section Usage with `compile`:
#' ```r
#' model %>% compile(
#'   optimizer = 'sgd',
#'   loss = 'mse',
#'   metrics = list(metric_SOME_METRIC(), metric_SOME_OTHER_METRIC())
#' )
#' ```
#'
#' @section Standalone usage:
#' ```r
#' m <- metric_SOME_METRIC()
#' for (e in seq(epochs)) {
#'   for (i in seq(train_steps)) {
#'     c(y_true, y_pred, sample_weight = NULL) %<-% ...
#'     m$update_state(y_true, y_pred, sample_weight)
#'   }
#'   cat('Final epoch result: ', as.numeric(m$result()), "\n")
#'   m$reset_state()
#' }
#' ```
#'
#' @section Custom Metric (subclass):
#' To be implemented by subclasses:
#'
#'   *  `initialize()`: All state variables should be created in this method by calling `self$add_weight()` like:
#'
#'          self$var <- self$add_weight(...)
#'
#'   *  `update_state()`: Has all updates to the state variables like:
#'
#'          self$var$assign_add(...)
#'
#'   *  `result()`: Computes and returns a value for the metric from the state variables.
#'
#' Example custom metric subclass:
#' ````R
#' metric_binary_true_positives <- new_metric_class(
#'   classname = "BinaryTruePositives",
#'   initialize = function(name = 'binary_true_positives', ...) {
#'     super$initialize(name = name, ...)
#'     self$true_positives <-
#'       self$add_weight(name = 'tp', initializer = 'zeros')
#'   },
#'
#'   update_state = function(y_true, y_pred, sample_weight = NULL) {
#'     y_true <- k_cast(y_true, "bool")
#'     y_pred <- k_cast(y_pred, "bool")
#'
#'     values <- y_true & y_pred
#'     values <- k_cast(values, self$dtype)
#'     if (!is.null(sample_weight)) {
#'       sample_weight <- k_cast(sample_weight, self$dtype)
#'       sample_weight <- tf$broadcast_to(sample_weight, values$shape)
#'       values <- values * sample_weight
#'     }
#'     self$true_positives$assign_add(tf$reduce_sum(values))
#'   },
#'
#'   result = function()
#'     self$true_positives
#' )
#' model %>% compile(..., metrics = list(metric_binary_true_positives()))
#' ````
#' The same `metric_binary_true_positives` could be built with `%py_class%` like
#' this:
#' ````
#' metric_binary_true_positives(keras$metrics$Metric) %py_class% {
#'   initialize <- <same-as-above>,
#'   update_state <- <same-as-above>,
#'   result <- <same-as-above>
#' }
#' ````
#'
#' @name Metric
#' @rdname Metric
NULL


#' @title metric-or-Metric
#' @name metric-or-Metric
#' @rdname metric-or-Metric
#' @keywords internal
#'
#' @param y_true Tensor of true targets.
#' @param y_pred Tensor of predicted targets.
#' @param ... Passed on to the underlying metric. Used for forwards and backwards compatibility.
#' @param axis (Optional) (1-based) Defaults to -1. The dimension along which the metric is computed.
#' @param name (Optional) string name of the metric instance.
#' @param dtype (Optional) data type of the metric result.
#'
#' @returns If `y_true` and `y_pred` are missing, a (subclassed) `Metric`
#'   instance is returned. The `Metric` object can be passed directly to
#'   `compile(metrics = )` or used as a standalone object. See `?Metric` for
#'   example usage.
#'
#'   Alternatively, if called with `y_true` and `y_pred` arguments, then the
#'   computed case-wise values for the mini-batch are returned directly.
NULL


# if(!exists("isFALSE"))
if(getRversion() < "3.5")
  isFALSE <- function(x) {
    is.logical(x) && length(x) == 1L && !is.na(x) && !x
  }

py_metric_wrapper <- function(py_fn, py_cls, formals=NULL, modifiers=NULL,
                           py_fn_name = TRUE) {
  modifiers <- substitute(modifiers)
  py_fn <- substitute(py_fn)
  py_cls <- substitute(py_cls)

  if(is.symbol(py_cls))
    py_cls <- substitute(keras$metrics$py_cls)

  if(is.symbol(py_fn))
    py_fn <- substitute(keras$metrics$py_fn)

  if("axis" %in% names(formals))
    modifiers$axis <- quote(as_axis)


  if (is.null(py_fn)) {
    body <- substitute({
      args <- capture_args(match.call(), modifiers)
      do.call(py_cls, args)
    })

    formals <- c(alist(... =), formals)
    if (!is.character(py_fn_name))
      py_fn_name <- NULL

  } else {

    body <- substitute({
      args <- capture_args(match.call(), modifiers)
      py_callable <- if (missing(y_true) && missing(y_pred))
        py_cls else py_fn
      do.call(py_callable, args)
    })

    formals <- c(alist(y_true = , y_pred =), formals)
    if (!isFALSE(py_fn_name)) {
      py_fn_name <- if (isTRUE(py_fn_name)) {
        last <- function(x) x[[length(x)]]
        last(strsplit(deparse(py_fn), "$", fixed = TRUE)[[1]])
      }
      else
        NULL
    }
  }

  formals[["..."]] <- quote(expr = )

  if (!is.null(py_cls)) {
    if (!"name" %in% names(formals))
      formals['name'] <- list(py_fn_name)
    if (!"dtype" %in% names(formals))
      formals['dtype'] <- list(NULL)
  }

  fn <- as.function.default(c(formals, body), envir = parent.frame())

  if(is.character(py_fn_name))
    attr(fn, "py_function_name") <- py_fn_name

  fn
}







#' Approximates the AUC (Area under the curve) of the ROC or PR curves
#'
#' @details The AUC (Area under the curve) of the ROC (Receiver operating
#' characteristic; default) or PR (Precision Recall) curves are quality measures
#' of binary classifiers. Unlike the accuracy, and like cross-entropy losses,
#' ROC-AUC and PR-AUC evaluate all the operational points of a model.
#'
#' This class approximates AUCs using a Riemann sum. During the metric
#' accumulation phrase, predictions are accumulated within predefined buckets by
#' value. The AUC is then computed by interpolating per-bucket averages. These
#' buckets define the evaluated operational points.
#'
#' This metric creates four local variables, `true_positives`, `true_negatives`,
#' `false_positives` and `false_negatives` that are used to compute the AUC. To
#' discretize the AUC curve, a linearly spaced set of thresholds is used to
#' compute pairs of recall and precision values. The area under the ROC-curve is
#' therefore computed using the height of the recall values by the false
#' positive rate, while the area under the PR-curve is the computed using the
#' height of the precision values by the recall.
#'
#' This value is ultimately returned as `auc`, an idempotent operation that
#' computes the area under a discretized curve of precision versus recall values
#' (computed using the aforementioned variables). The `num_thresholds` variable
#' controls the degree of discretization with larger numbers of thresholds more
#' closely approximating the true AUC. The quality of the approximation may vary
#' dramatically depending on `num_thresholds`. The `thresholds` parameter can be
#' used to manually specify thresholds which split the predictions more evenly.
#'
#' For a best approximation of the real AUC, `predictions` should be distributed
#' approximately uniformly in the range `[0, 1]` (if `from_logits=FALSE`). The
#' quality of the AUC approximation may be poor if this is not the case. Setting
#' `summation_method` to 'minoring' or 'majoring' can help quantify the error in
#' the approximation by providing lower or upper bound estimate of the AUC.
#'
#' If `sample_weight` is `NULL`, weights default to 1. Use `sample_weight` of 0
#' to mask values.
#'
#' @param num_thresholds (Optional) Defaults to 200. The number of thresholds toa
#'   use when discretizing the roc curve. Values must be > 1.
#'
#' @param curve (Optional) Specifies the name of the curve to be computed, 'ROC'
#'   (default) or 'PR' for the Precision-Recall-curve.
#'
#' @param summation_method (Optional) Specifies the [Riemann summation method](
#'   https://en.wikipedia.org/wiki/Riemann_sum) used. 'interpolation' (default)
#'   applies mid-point summation scheme for `ROC`. For PR-AUC, interpolates
#'   (true/false) positives but not the ratio that is precision (see Davis &
#'   Goadrich 2006 for details); 'minoring' applies left summation for
#'   increasing intervals and right summation for decreasing intervals;
#'   'majoring' does the opposite.
#'
#' @param thresholds (Optional) A list of floating point values to use as the
#'   thresholds for discretizing the curve. If set, the `num_thresholds`
#'   parameter is ignored. Values should be in `[0, 1]`. Endpoint thresholds equal
#'   to {-epsilon, 1+epsilon} for a small positive epsilon value will be
#'   automatically included with these to correctly handle predictions equal to
#'   exactly 0 or 1.
#'
#' @param multi_label boolean indicating whether multilabel data should be
#'   treated as such, wherein AUC is computed separately for each label and then
#'   averaged across labels, or (when FALSE) if the data should be flattened
#'   into a single label before AUC computation. In the latter case, when
#'   multilabel data is passed to AUC, each label-prediction pair is treated as
#'   an individual data point. Should be set to FALSE for multi-class data.
#'
#' @param num_labels (Optional) The number of labels, used when `multi_label` is
#'   TRUE. If `num_labels` is not specified, then state variables get created on
#'   the first call to `update_state`.
#'
#' @param label_weights (Optional) list, array, or tensor of non-negative
#'   weights used to compute AUCs for multilabel data. When `multi_label` is
#'   TRUE, the weights are applied to the individual label AUCs when they are
#'   averaged to produce the multi-label AUC. When it's FALSE, they are used to
#'   weight the individual label predictions in computing the confusion matrix
#'   on the flattened data. Note that this is unlike class_weights in that
#'   class_weights weights the example depending on the value of its label,
#'   whereas label_weights depends only on the index of that label before
#'   flattening; therefore `label_weights` should not be used for multi-class
#'   data.
#'
#' @param from_logits boolean indicating whether the predictions (`y_pred` in
#'   `update_state`) are probabilities or sigmoid logits. As a rule of thumb,
#'   when using a keras loss, the `from_logits` constructor argument of the loss
#'   should match the AUC `from_logits` constructor argument.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_auc <- py_metric_wrapper(
  NULL, AUC,
  alist(
    num_thresholds = 200L,
    curve = 'ROC',
    summation_method = 'interpolation',
    thresholds = NULL,
    multi_label = FALSE,
    num_labels = NULL,
    label_weights = NULL,
    from_logits = FALSE
  ),
  list(num_thresholds = as.integer)
)



#' Calculates how often predictions equal labels
#'
#' @details
#' This metric creates two local variables, `total` and `count` that are used to
#' compute the frequency with which `y_pred` matches `y_true`. This frequency is
#' ultimately returned as `binary accuracy`: an idempotent operation that simply
#' divides `total` by `count`.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_accuracy <- py_metric_wrapper(
  NULL, Accuracy
)



#' Computes the recall of the predictions with respect to the labels
#'
#' @details This metric creates two local variables, `true_positives` and
#' `false_negatives`, that are used to compute the recall. This value is
#' ultimately returned as `recall`, an idempotent operation that simply divides
#' `true_positives` by the sum of `true_positives` and `false_negatives`.
#'
#' If `sample_weight` is `NULL`, weights default to 1. Use `sample_weight` of 0
#' to mask values.
#'
#' If `top_k` is set, recall will be computed as how often on average a class
#' among the labels of a batch entry is in the top-k predictions.
#'
#' If `class_id` is specified, we calculate recall by considering only the
#' entries in the batch for which `class_id` is in the label, and computing the
#' fraction of them for which `class_id` is above the threshold and/or in the
#' top-k predictions.
#'
#' @param thresholds (Optional) A float value or a list of float
#'   threshold values in `[0, 1]`. A threshold is compared with prediction values
#'   to determine the truth value of predictions (i.e., above the threshold is
#'   `true`, below is `false`). One metric value is generated for each threshold
#'   value. If neither thresholds nor top_k are set, the default is to calculate
#'   recall with `thresholds=0.5`.
#'
#' @param top_k (Optional) Unset by default. An int value specifying the top-k
#'   predictions to consider when calculating recall.
#'
#' @param class_id (Optional) Integer class ID for which we want binary metrics.
#'   This must be in the half-open interval `[0, num_classes)`, where
#'   `num_classes` is the last dimension of predictions.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_recall <- py_metric_wrapper(
  NULL, Recall,
  alist(thresholds=NULL, top_k=NULL, class_id=NULL),
  list(top_k = as_nullable_integer,
       class_id = as_nullable_integer)
)



#' Computes best recall where precision is >= specified value
#'
#' @details For a given score-label-distribution the required precision might
#' not be achievable, in this case 0.0 is returned as recall.
#'
#' This metric creates four local variables, `true_positives`, `true_negatives`,
#' `false_positives` and `false_negatives` that are used to compute the recall
#' at the given precision. The threshold for the given precision value is
#' computed and used to evaluate the corresponding recall.
#'
#' If `sample_weight` is `NULL`, weights default to 1. Use `sample_weight` of 0
#' to mask values.
#'
#' If `class_id` is specified, we calculate precision by considering only the
#' entries in the batch for which `class_id` is above the threshold predictions,
#' and computing the fraction of them for which `class_id` is indeed a correct
#' label.
#'
#' @param precision A scalar value in range `[0, 1]`.
#'
#' @param num_thresholds (Optional) Defaults to 200. The number of thresholds to
#'   use for matching the given precision.
#'
#' @param class_id (Optional) Integer class ID for which we want binary metrics.
#'   This must be in the half-open interval `[0, num_classes)`, where
#'   `num_classes` is the last dimension of predictions.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_recall_at_precision <- py_metric_wrapper(
  NULL, RecallAtPrecision,
  alist(precision=, num_thresholds=200L, class_id=NULL),
  list(num_thresholds = as.integer)
)



#' Computes the precision of the predictions with respect to the labels
#'
#' @details The metric creates two local variables, `true_positives` and
#' `false_positives` that are used to compute the precision. This value is
#' ultimately returned as `precision`, an idempotent operation that simply
#' divides `true_positives` by the sum of `true_positives` and
#' `false_positives`.
#'
#' If `sample_weight` is `NULL`, weights default to 1. Use `sample_weight` of 0
#' to mask values.
#'
#' If `top_k` is set, we'll calculate precision as how often on average a class
#' among the top-k classes with the highest predicted values of a batch entry is
#' correct and can be found in the label for that entry.
#'
#' If `class_id` is specified, we calculate precision by considering only the
#' entries in the batch for which `class_id` is above the threshold and/or in
#' the top-k highest predictions, and computing the fraction of them for which
#' `class_id` is indeed a correct label.
#'
#' @param thresholds (Optional) A float value or a list of float
#'   threshold values in `[0, 1]`. A threshold is compared with prediction values
#'   to determine the truth value of predictions (i.e., above the threshold is
#'   `true`, below is `false`). One metric value is generated for each threshold
#'   value. If neither thresholds nor top_k are set, the default is to calculate
#'   precision with `thresholds=0.5`.
#'
#' @param top_k (Optional) Unset by default. An int value specifying the top-k
#'   predictions to consider when calculating precision.
#'
#' @param class_id (Optional) Integer class ID for which we want binary metrics.
#'   This must be in the half-open interval `[0, num_classes)`, where
#'   `num_classes` is the last dimension of predictions.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_precision <- py_metric_wrapper(
  NULL, Precision,
  alist(thresholds=NULL, top_k=NULL, class_id=NULL),
  list(top_k = as_nullable_integer)
)



#' Computes best precision where recall is >= specified value
#'
#' @details This metric creates four local variables, `true_positives`,
#' `true_negatives`, `false_positives` and `false_negatives` that are used to
#' compute the precision at the given recall. The threshold for the given recall
#' value is computed and used to evaluate the corresponding precision.
#'
#' If `sample_weight` is `NULL`, weights default to 1. Use `sample_weight` of 0
#' to mask values.
#'
#' If `class_id` is specified, we calculate precision by considering only the
#' entries in the batch for which `class_id` is above the threshold predictions,
#' and computing the fraction of them for which `class_id` is indeed a correct
#' label.
#'
#' @param recall A scalar value in range `[0, 1]`.
#'
#' @param num_thresholds (Optional) Defaults to 200. The number of thresholds to
#'   use for matching the given recall.
#'
#' @param class_id (Optional) Integer class ID for which we want binary metrics.
#'   This must be in the half-open interval `[0, num_classes)`, where
#'   `num_classes` is the last dimension of predictions.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_precision_at_recall <- py_metric_wrapper(
  NULL, PrecisionAtRecall,
  alist(recall=, num_thresholds=200L, class_id=NULL),
  list(num_thresholds = as.integer)
)



#' Computes root mean squared error metric between `y_true` and `y_pred`
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_root_mean_squared_error <- py_metric_wrapper(
  NULL, RootMeanSquaredError
)



#' Computes best sensitivity where specificity is >= specified value
#'
#' The sensitivity at a given specificity.
#'
#' `Sensitivity` measures the proportion of actual positives that are correctly
#' identified as such `(tp / (tp + fn))`. `Specificity` measures the proportion of
#' actual negatives that are correctly identified as such `(tn / (tn + fp))`.
#'
#' This metric creates four local variables, `true_positives`, `true_negatives`,
#' `false_positives` and `false_negatives` that are used to compute the
#' sensitivity at the given specificity. The threshold for the given specificity
#' value is computed and used to evaluate the corresponding sensitivity.
#'
#' If `sample_weight` is `NULL`, weights default to 1. Use `sample_weight` of 0
#' to mask values.
#'
#' If `class_id` is specified, we calculate precision by considering only the
#' entries in the batch for which `class_id` is above the threshold predictions,
#' and computing the fraction of them for which `class_id` is indeed a correct
#' label.
#'
#' For additional information about specificity and sensitivity, see [the
#' following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).
#'
#' @param specificity A scalar value in range `[0, 1]`.
#'
#' @param num_thresholds (Optional) Defaults to 200. The number of thresholds to
#'   use for matching the given specificity.
#'
#' @param class_id (Optional) Integer class ID for which we want binary metrics.
#'   This must be in the half-open interval `[0, num_classes)`, where
#'   `num_classes` is the last dimension of predictions.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_sensitivity_at_specificity <- py_metric_wrapper(
  NULL, SensitivityAtSpecificity,
  alist(specificity = , num_thresholds = 200L, class_id = NULL),
  list(num_thresholds = as.integer)
)



#' Computes best specificity where sensitivity is >= specified value
#'
#' @details
#' `Sensitivity` measures the proportion of actual positives that are correctly
#' identified as such `(tp / (tp + fn))`.
#' `Specificity` measures the proportion of actual negatives that are correctly
#' identified as such `(tn / (tn + fp))`.
#'
#' This metric creates four local variables, `true_positives`, `true_negatives`,
#' `false_positives` and `false_negatives` that are used to compute the
#' specificity at the given sensitivity. The threshold for the given sensitivity
#' value is computed and used to evaluate the corresponding specificity.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' If `class_id` is specified, we calculate precision by considering only the
#' entries in the batch for which `class_id` is above the threshold predictions,
#' and computing the fraction of them for which `class_id` is indeed a correct
#' label.
#'
#' For additional information about specificity and sensitivity, see
#' [the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).
#'
#' @param sensitivity A scalar value in range `[0, 1]`.
#'
#' @param num_thresholds (Optional) Defaults to 200. The number of thresholds to
#' use for matching the given sensitivity.
#'
#' @param class_id (Optional) Integer class ID for which we want binary metrics.
#' This must be in the half-open interval `[0, num_classes)`, where
#' `num_classes` is the last dimension of predictions.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_specificity_at_sensitivity <- py_metric_wrapper(
  NULL, SpecificityAtSensitivity,
  alist(sensitivity = , num_thresholds = 200L, class_id = NULL),
  list(num_thresholds = as.integer)
)



#' Computes the (weighted) sum of the given values
#'
#' @details
#' For example, if values is `c(1, 3, 5, 7)` then the sum is 16.
#' If the weights were specified as `c(1, 1, 0, 0)` then the sum would be 4.
#'
#' This metric creates one variable, `total`, that is used to compute the sum of
#' `values`. This is ultimately returned as `sum`.
#'
#' If `sample_weight` is `NULL`, weights default to 1.  Use `sample_weight` of 0
#' to mask values.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_sum <-  py_metric_wrapper(
  NULL, Sum
)



#' Calculates how often predictions match binary labels
#'
#' @details
#' This metric creates two local variables, `total` and `count` that are used to
#' compute the frequency with which `y_pred` matches `y_true`. This frequency is
#' ultimately returned as `binary accuracy`: an idempotent operation that simply
#' divides `total` by `count`.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' @param threshold (Optional) Float representing the threshold for deciding
#'   whether prediction values are 1 or 0.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit metric-or-Metric return
#' @family metrics
#' @export
metric_binary_accuracy <- py_metric_wrapper(
  binary_accuracy, BinaryAccuracy,
  alist(threshold = 0.5)
)



#' Computes the crossentropy metric between the labels and predictions
#'
#' @details
#' This is the crossentropy metric class to be used when there are only two
#' label classes (0 and 1).
#'
#' @param from_logits (Optional) Whether output is expected to be a logits tensor.
#' By default, we consider that output encodes a probability distribution.
#'
#' @param label_smoothing (Optional) Float in `[0, 1]`. When > 0, label values are
#' smoothed, meaning the confidence on label values are relaxed.
#' e.g. `label_smoothing = 0.2` means that we will use a value of `0.1` for
#' label `0` and `0.9` for label `1`".
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit metric-or-Metric return
#' @family metrics
#' @export
metric_binary_crossentropy <- py_metric_wrapper(
  binary_crossentropy, BinaryCrossentropy,
  alist(from_logits=FALSE, label_smoothing=0, axis=-1L)
)



#' Calculates how often predictions match one-hot labels
#'
#' @details
#' You can provide logits of classes as `y_pred`, since argmax of
#' logits and probabilities are same.
#'
#' This metric creates two local variables, `total` and `count` that are used to
#' compute the frequency with which `y_pred` matches `y_true`. This frequency is
#' ultimately returned as `categorical accuracy`: an idempotent operation that
#' simply divides `total` by `count`.
#'
#' `y_pred` and `y_true` should be passed in as vectors of probabilities, rather
#' than as labels. If necessary, use `tf.one_hot` to expand `y_true` as a vector.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit metric-or-Metric return
#' @family metrics
#' @export
metric_categorical_accuracy <- py_metric_wrapper(
  categorical_accuracy, CategoricalAccuracy
)



#' Computes the crossentropy metric between the labels and predictions
#'
#' @details
#' This is the crossentropy metric class to be used when there are multiple
#' label classes (2 or more). Here we assume that labels are given as a `one_hot`
#' representation. eg., When labels values are `c(2, 0, 1)`:
#' ```
#'  y_true = rbind(c(0, 0, 1),
#'                 c(1, 0, 0),
#'                 c(0, 1, 0))`
#' ```
#' @param from_logits (Optional) Whether output is expected to be a logits tensor.
#' By default, we consider that output encodes a probability distribution.
#'
#' @param label_smoothing (Optional) Float in `[0, 1]`. When > 0, label values are
#' smoothed, meaning the confidence on label values are relaxed. e.g.
#' `label_smoothing=0.2` means that we will use a value of `0.1` for label
#' `0` and `0.9` for label `1`"
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit metric-or-Metric return
#' @family metrics
#' @export
metric_categorical_crossentropy <- py_metric_wrapper(
  categorical_crossentropy, CategoricalCrossentropy,
  alist(from_logits = FALSE, label_smoothing = 0, axis = -1L)
)



#' Computes the cosine similarity between the labels and predictions
#'
#' @details
#' ```
#' cosine similarity = (a . b) / ||a|| ||b||
#' ```
#'
#' See: [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
#'
#' This metric keeps the average cosine similarity between `predictions` and
#' `labels` over a stream of data.
#'
#' @note If you want to compute the cosine_similarity for each case in a
#' mini-batch you can use `loss_cosine_similarity()`.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_cosine_similarity <- py_metric_wrapper(
  NULL,
  CosineSimilarity,
  alist(axis=-1L, name='cosine_similarity')
)



#' Calculates the number of false negatives
#'
#' @details
#' If `sample_weight` is given, calculates the sum of the weights of
#' false negatives. This metric creates one local variable, `accumulator`
#' that is used to keep track of the number of false negatives.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' @param thresholds (Optional) Defaults to 0.5. A float value or a
#' list of float threshold values in `[0, 1]`. A threshold is compared
#' with prediction values to determine the truth value of predictions
#' (i.e., above the threshold is `TRUE`, below is `FALSE`). One metric
#' value is generated for each threshold value.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_false_negatives <- py_metric_wrapper(
  NULL, FalseNegatives,
  alist(thresholds = NULL))



#' Calculates the number of false positives
#'
#' @details
#' If `sample_weight` is given, calculates the sum of the weights of
#' false positives. This metric creates one local variable, `accumulator`
#' that is used to keep track of the number of false positives.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' @param thresholds (Optional) Defaults to 0.5. A float value or a
#' list of float threshold values in `[0, 1]`. A threshold is compared
#' with prediction values to determine the truth value of predictions
#' (i.e., above the threshold is `true`, below is `false`). One metric
#' value is generated for each threshold value.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_false_positives <- py_metric_wrapper(
  NULL, FalsePositives,
  alist(thresholds = NULL))



#' Calculates the number of true negatives
#'
#' @details
#' If `sample_weight` is given, calculates the sum of the weights of
#' true negatives. This metric creates one local variable, `accumulator`
#' that is used to keep track of the number of true negatives.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' @param thresholds (Optional) Defaults to 0.5. A float value or a
#' list of float threshold values in `[0, 1]`. A threshold is compared
#' with prediction values to determine the truth value of predictions
#' (i.e., above the threshold is `true`, below is `false`). One metric
#' value is generated for each threshold value.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_true_negatives <- py_metric_wrapper(
  NULL, TrueNegatives,
  alist(thresholds = NULL))



#' Calculates the number of true positives
#'
#' @details
#' If `sample_weight` is given, calculates the sum of the weights of
#' true positives. This metric creates one local variable, `true_positives`
#' that is used to keep track of the number of true positives.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' @param thresholds (Optional) Defaults to 0.5. A float value or a
#' list of float threshold values in `[0, 1]`. A threshold is compared
#' with prediction values to determine the truth value of predictions
#' (i.e., above the threshold is `true`, below is `false`). One metric
#' value is generated for each threshold value.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_true_positives <- py_metric_wrapper(
  NULL, TruePositives,
  alist(thresholds = NULL))



#' Computes the hinge metric between `y_true` and `y_pred`
#'
#' `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
#' provided we will convert them to -1 or 1.
#'
#' ```
#' loss = tf$reduce_mean(tf$maximum(1 - y_true * y_pred, 0L), axis=-1L)
#' ```
#'
#' @inheritParams metric-or-Metric
#' @inherit metric-or-Metric return
#' @family metrics
#' @export
metric_hinge <- py_metric_wrapper(hinge, Hinge)


#' Computes Kullback-Leibler divergence
#'
#' @details
#' ```
#' metric = y_true * log(y_true / y_pred)
#' ```
#'
#' See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit metric-or-Metric return
#' @family metrics
#' @export
metric_kullback_leibler_divergence <- py_metric_wrapper(
  kullback_leibler_divergence, KLDivergence
)



#' Computes the logarithm of the hyperbolic cosine of the prediction error
#'
#' `logcosh = log((exp(x) + exp(-x))/2)`, where x is the error (`y_pred - y_true`)
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_logcosh_error <- py_metric_wrapper(
  NULL, LogCoshError,
  alist(name = "logcosh")
)



#' Computes the (weighted) mean of the given values
#'
#' @details
#' For example, if values is `c(1, 3, 5, 7)` then the mean is 4.
#' If the weights were specified as `c(1, 1, 0, 0)` then the mean would be 2.
#'
#' This metric creates two variables, `total` and `count` that are used to
#' compute the average of `values`. This average is ultimately returned as `mean`
#' which is an idempotent operation that simply divides `total` by `count`.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' @note Unlike most other metrics, this only takes a single tensor as input to update state.
#'
#' Example usage with `compile()`:
#' ````
#' model$add_metric(metric_mean(name='mean_1')(outputs))
#' model %>% compile(optimizer='sgd', loss='mse')
#' ````
#' Example standalone usage:
#' ```
#' m  <- metric_mean()
#' m$update_state(c(1, 3, 5, 7))
#' m$result()
#'
#' m$reset_state()
#' m$update_state(c(1, 3, 5, 7), sample_weight=c(1, 1, 0, 0))
#' m$result()
#' as.numeric(m$result())
#' ```
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_mean <- py_metric_wrapper(
  NULL, Mean,
  alist(name = "mean")
)



#' Computes the mean absolute error between the labels and predictions
#'
#' @details
#' `loss = mean(abs(y_true - y_pred), axis=-1)`
#'
#' @inheritParams metric-or-Metric
#' @inherit metric-or-Metric return
#' @family metrics
#' @export
metric_mean_absolute_error <- py_metric_wrapper(
  mean_absolute_error, MeanAbsoluteError
)



#' Computes the mean absolute percentage error between `y_true` and `y_pred`
#'
#' @details
#' `loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)`
#'
#' @inheritParams metric-or-Metric
#' @inherit metric-or-Metric return
#' @family metrics
#' @export
metric_mean_absolute_percentage_error <- py_metric_wrapper(
  mean_absolute_percentage_error, MeanAbsolutePercentageError
)



#' Computes the mean Intersection-Over-Union metric
#'
#' @details
#' Mean Intersection-Over-Union is a common evaluation metric for semantic image
#' segmentation, which first computes the IOU for each semantic class and then
#' computes the average over classes. IOU is defined as follows:
#' ````
#'   IOU = true_positive / (true_positive + false_positive + false_negative)
#' ````
#' The predictions are accumulated in a confusion matrix, weighted by
#' `sample_weight` and the metric is then calculated from it.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' @param num_classes The possible number of labels the prediction task can have.
#' This value must be provided, since a confusion matrix of `dim`
#' `c(num_classes, num_classes)` will be allocated.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_mean_iou <- py_metric_wrapper(
  NULL, MeanIoU,
  alist(num_classes = ),
  list(num_classes = as.integer)
)



#' Wraps a stateless metric function with the Mean metric
#'
#' @details
#' You could use this class to quickly build a mean metric from a function. The
#' function needs to have the signature `fn(y_true, y_pred)` and return a
#' per-sample loss array. `MeanMetricWrapper$result()` will return
#' the average metric value across all samples seen so far.
#'
#' For example:
#'
#' ```r
#' accuracy <- function(y_true, y_pred)
#'   k_cast(y_true == y_pred, 'float32')
#'
#' accuracy_metric <- metric_mean_wrapper(fn = accuracy)
#'
#' model %>% compile(..., metrics=accuracy_metric)
#' ```
#'
#' @param fn The metric function to wrap, with signature `fn(y_true, y_pred, ...)`.
#'
#' @param ... named arguments to pass on to `fn`.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_mean_wrapper <- py_metric_wrapper(
  NULL, MeanMetricWrapper,
  alist(fn = )
)



#' Computes the mean relative error by normalizing with the given values
#'
#' @details
#' This metric creates two local variables, `total` and `count` that are used to
#' compute the mean relative error. This is weighted by `sample_weight`, and
#' it is ultimately returned as `mean_relative_error`:
#' an idempotent operation that simply divides `total` by `count`.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' ```
#' metric = mean(|y_pred - y_true| / normalizer)
#' ```
#' For example:
#' ```
#' m = metric_mean_relative_error(normalizer=c(1, 3, 2, 3))
#' m$update_state(c(1, 3, 2, 3), c(2, 4, 6, 8))
#'  # result     = mean(c(1, 1, 4, 5) / c(1, 3, 2, 3)) = mean(c(1, 1/3, 2, 5/3))
#'  #            = 5/4 = 1.25
#' m$result()
#' ```
#'
#' @param normalizer The normalizer values with same shape as predictions.
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_mean_relative_error <- py_metric_wrapper(
  NULL, MeanRelativeError,
  alist(normalizer = )
)



#' Computes the mean squared error between labels and predictions
#'
#' @details
#' After computing the squared distance between the inputs, the mean value over
#' the last dimension is returned.
#'
#' `loss = mean(square(y_true - y_pred), axis=-1)`
#'
#' @inheritParams metric-or-Metric
#' @inherit metric-or-Metric return
#' @family metrics
#' @export
metric_mean_squared_error <- py_metric_wrapper(
  mean_absolute_percentage_error, MeanAbsolutePercentageError
)



#' Computes the mean squared logarithmic error
#'
#' @details
#' `loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)`
#'
#' @inheritParams metric-or-Metric
#' @inherit metric-or-Metric return
#' @family metrics
#' @export
metric_mean_squared_logarithmic_error <- py_metric_wrapper(
  mean_squared_logarithmic_error, MeanSquaredLogarithmicError
)



#' Computes the element-wise (weighted) mean of the given tensors
#'
#' @details
#' `MeanTensor` returns a tensor with the same shape of the input tensors. The
#' mean value is updated by keeping local variables `total` and `count`. The
#' `total` tracks the sum of the weighted values, and `count` stores the sum of
#' the weighted counts.
#'
#' @param shape (Optional) A list of integers, a list of integers, or a 1-D Tensor
#' of type int32. If not specified, the shape is inferred from the values at
#' the first call of update_state.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_mean_tensor <- py_metric_wrapper(
  NULL, MeanTensor,
  alist(shape = NULL)
)



#' Computes the Poisson metric between `y_true` and `y_pred`
#'
#' `metric = y_pred - y_true * log(y_pred)`
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit metric-or-Metric return
#' @family metrics
#' @export
metric_poisson <- py_metric_wrapper(
  poisson, Poisson
)



#' Computes the crossentropy metric between the labels and predictions
#'
#' @details
#' Use this crossentropy metric when there are two or more label classes.
#' We expect labels to be provided as integers. If you want to provide labels
#' using `one-hot` representation, please use `CategoricalCrossentropy` metric.
#' There should be `# classes` floating point values per feature for `y_pred`
#' and a single floating point value per feature for `y_true`.
#'
#' In the snippet below, there is a single floating point value per example for
#' `y_true` and `# classes` floating pointing values per example for `y_pred`.
#' The shape of `y_true` is `[batch_size]` and the shape of `y_pred` is
#' `[batch_size, num_classes]`.
#'
#' @param from_logits (Optional) Whether output is expected to be a logits tensor.
#' By default, we consider that output encodes a probability distribution.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit metric-or-Metric return
#' @family metrics
#' @export
metric_sparse_categorical_crossentropy <- py_metric_wrapper(
  sparse_categorical_crossentropy, SparseCategoricalCrossentropy,
  alist(from_logits=FALSE, axis = -1L)
)



#' Computes the squared hinge metric
#'
#' `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
#' provided we will convert them to -1 or 1.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit metric-or-Metric return
#' @family metrics
#' @export
metric_squared_hinge <- py_metric_wrapper(
  squared_hinge, SquaredHinge
)



#' Computes the categorical hinge metric between `y_true` and `y_pred`
#'
#' @inheritParams metric-or-Metric
#' @inherit Metric return
#' @family metrics
#' @export
metric_categorical_hinge <- py_metric_wrapper(
  NULL, CategoricalHinge
)



#' Calculates how often predictions match integer labels
#'
#' @details
#' ```r
#' acc = k_dot(sample_weight, y_true == k_argmax(y_pred, axis=2))
#' ```
#'
#' You can provide logits of classes as `y_pred`, since argmax of
#' logits and probabilities are same.
#'
#' This metric creates two local variables, `total` and `count` that are used to
#' compute the frequency with which `y_pred` matches `y_true`. This frequency is
#' ultimately returned as `sparse categorical accuracy`: an idempotent operation
#' that simply divides `total` by `count`.
#'
#' If `sample_weight` is `NULL`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit metric-or-Metric return
#' @family metrics
#' @export
metric_sparse_categorical_accuracy <- py_metric_wrapper(
  sparse_categorical_accuracy, SparseCategoricalAccuracy
)



#' Computes how often targets are in the top `K` predictions
#'
#'
#' @param k (Optional) Number of top elements to look at for computing accuracy.
#' Defaults to 5.
#'
#'
#' @inheritParams metric-or-Metric
#' @inherit metric-or-Metric return
#' @family metrics
#' @export
metric_top_k_categorical_accuracy <- py_metric_wrapper(
  top_k_categorical_accuracy, TopKCategoricalAccuracy,
  alist(k=5L),
  list(k=as.integer)
)



#' Computes how often integer targets are in the top `K` predictions
#'
#' @param k (Optional) Number of top elements to look at for computing accuracy.
#' Defaults to 5.
#'
#' @inheritParams metric-or-Metric
#' @inherit metric-or-Metric return
#' @family metrics
#' @export
metric_sparse_top_k_categorical_accuracy <- py_metric_wrapper(
  sparse_top_k_categorical_accuracy, SparseTopKCategoricalAccuracy,
  alist(k=5L),
  list(k=as.integer)
)


#' Custom metric function
#'
#' @param name name used to show training progress output
#' @param metric_fn An R function with signature `function(y_true, y_pred){}` that accepts tensors.
#'
#' @details
#' You can provide an arbitrary R function as a custom metric. Note that
#' the `y_true` and `y_pred` parameters are tensors, so computations on
#' them should use backend tensor functions.
#'
#' Use the `custom_metric()` function to define a custom metric.
#' Note that a name ('mean_pred') is provided for the custom metric
#' function: this name is used within training progress output.
#'
#' If you want to save and load a model with custom metrics, you should
#' also specify the metric in the call the [load_model_hdf5()]. For example:
#' `load_model_hdf5("my_model.h5", c('mean_pred' = metric_mean_pred))`.
#'
#' Alternatively, you can wrap all of your code in a call to
#' [with_custom_object_scope()] which will allow you to refer to the
#' metric by name just like you do with built in keras metrics.
#'
#' Documentation on the available backend tensor functions can be
#' found at <https://tensorflow.rstudio.com/reference/keras/#backend>.
#'
#' Alternative ways of supplying custom metrics:
#'  +  `custom_metric():` Arbitrary R function.
#'  +  [metric_mean_wrapper()]: Wrap an arbitrary R function in a `Metric` instance.
#'  +  subclass `keras$metrics$Metric`: see `?Metric` for example.
#'
#' @family metrics
#' @export
custom_metric <- function(name, metric_fn) {
  metric_fn <- reticulate::py_func(metric_fn)
  reticulate::py_set_attr(metric_fn, "__name__", name)
  metric_fn
}



#' (Deprecated) metric_cosine_proximity
#'
#' `metric_cosine_proximity()` is deprecated and will be removed in a future
#' version. Please update your code to use `metric_cosine_similarity()` if
#' possible. If you need the actual function and not a Metric object, (e.g,
#' because you are using the intermediate computed values in a custom training
#' loop before reduction), please use `loss_cosine_similarity()` or
#' `tensorflow::tf$compat$v1$keras$metrics$cosine_proximity()`
#'
#' @inheritParams metric-or-Metric
#' @keywords internal
#' @export
metric_cosine_proximity <- function(y_true, y_pred) {
  warning(
"metric_cosine_proximity() is deprecated and will be removed in a future version.",
" Please update your code to use metric_cosine_similarity() if possible.",
" If you need the actual function and not a Metric object,",
" (e.g, because you are using the intermediate computed values",
" in a custom training loop before reduction), please use loss_cosine_similarity() or",
" tensorflow::tf$compat$v1$keras$metrics$cosine_proximity()")
  tensorflow::tf$compat$v1$keras$metrics$cosine_proximity(y_true, y_pred)
}
attr(metric_cosine_proximity, "py_function_name") <- "cosine_proximity"





### some interactive snippets use to autogenerate the starters for docs above.
### There is still quite a bit of manual massaging the docs needed after this.
# library(tidyverse)
#
# inspect <- reticulate::import("inspect")
#
# docstring_parser <- reticulate::import("docstring_parser")
# # reticulate::py_install("docstring_parser", pip = TRUE)
#
# get_doc <- function(py_obj) {
#   doc <- docstring_parser$parse(
#     inspect$getdoc(py_obj))
#   doc$object <- py_obj
#   doc
#     # style = docstring_parser$DocstringStyle$GOOGLE)
#     # ## not all doc strings successfully parse google style,
#     # ## some default to REST style
# }
#
#
# py_str.docstring_parser.common.Docstring <- function(x) {
#   cat(docstring_parser$compose(x))
# }
#
#
# cleanup_description <- function(x) {
#
#     # remove leading and trailing whitespace
#     x <- gsub("^\\s+|\\s+$", "", x)
#
#     # convert 2+ whitespace to 1 ws
#     # x <- gsub("(\\s\\s+)", " ", x)
#
#     # convert literals
#     x <- gsub("None", "NULL", x, fixed=TRUE)
#     x <- gsub("True", "TRUE", x, fixed=TRUE)
#     x <- gsub("False", "FALSE", x, fixed=TRUE)
#
#     # convert tuple to list
#     x <- gsub("tuple", "list", x, fixed=TRUE)
#     x <- gsub("list/list", "list", x, fixed=TRUE)
#
#     x
# }
#
# as_metric_fn_doc <- function(x, name = NULL) {
#   con <- textConnection("r-doc", "w")
#   on.exit(close(con))
#   cat <- function(...,  file = con)
#     base::cat(..., "\n", file = file)
#
#   # first sentence is taken as title
#   # 2nd paragraph is taken as @description
#   # 3rd paragraph + is taken as @details
#   title <- cleanup_description(x$short_description)
#   # title should have no trailing '.'
#   if (str_sub(title, -1) == ".")
#     title <- str_sub(title, end = -2)
#
#   # cat("@title ", title)
#   cat(title)
#
#   desc <- cleanup_description(x$long_description)
#   cat()
#
#   # avoid splitting across @description and @details,
#   # so put everything in @details
#   if (length(desc) != 0 && str_detect(desc, "\n"))
#     cat("@details")
#   cat(desc)
#
#   for (p in x$params) {
#     if (p$arg_name %in% c("name", "dtype")) next
#     cat("\n@param", p$arg_name, cleanup_description(p$description))
#   }
#
#   cat()
#
#   cat("@inheritParams Metric")
#   cat("@inherit Metric return")
#   cat("@family metrics")
#   cat("@export")
#
#   x <- textConnectionValue(con)
#   x <- stringr::str_flatten(x, "\n")
#   x <- gsub("\n", "\n#' ", x)
#   x <- str_c("#' ", x, "\n", name)
#   x
# }
#
# x <- keras$metrics$AUC
# as_metric_fn_doc(get_doc(x)) %>% cat()
#
# if(!exists("scratch"))
# scrtch <- tempfile(fileext = ".R")
# keras$metrics %>%
#     names() %>%
#     grep("[A-Z]", ., value=TRUE) %>%
#     map(~as_metric_fn_doc(get_doc(keras$metrics[[.x]]), name = .x)) %>%
#     str_flatten(collapse = "\n\n\n") %>%
#     cat(file = scrtch)
#
# file.edit(scratch)
