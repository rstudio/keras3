

#' Computes the cross-entropy loss between true labels and predicted labels.
#'
#' @description
#' Use this cross-entropy loss for binary (0 or 1) classification applications.
#' The loss function requires the following inputs:
#'
#' - `y_true` (true label): This is either 0 or 1.
#' - `y_pred` (predicted value): This is the model's prediction, i.e, a single
#'     floating-point value which either represents a
#'     [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in `[-inf, inf]`
#'     when `from_logits=TRUE`) or a probability (i.e, value in `[0., 1.]` when
#'     `from_logits=FALSE`).
#'
#' # Examples
#' ```{r}
#' y_true <- rbind(c(0, 1), c(0, 0))
#' y_pred <- rbind(c(0.6, 0.4), c(0.4, 0.6))
#' loss <- loss_binary_crossentropy(y_true, y_pred)
#' loss
#' ```
#' **Recommended Usage:** (set `from_logits=TRUE`)
#'
#' With `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(
#'     loss = loss_binary_crossentropy(from_logits=TRUE),
#'     ...
#' )
#' ```
#'
#' As a standalone function:
#'
#' ```{r}
#' # Example 1: (batch_size = 1, number of samples = 4)
#' y_true <- op_array(c(0, 1, 0, 0))
#' y_pred <- op_array(c(-18.6, 0.51, 2.94, -12.8))
#' bce <- loss_binary_crossentropy(from_logits = TRUE)
#' bce(y_true, y_pred)
#' ```
#'
#' ```{r}
#' # Example 2: (batch_size = 2, number of samples = 4)
#' y_true <- rbind(c(0, 1), c(0, 0))
#' y_pred <- rbind(c(-18.6, 0.51), c(2.94, -12.8))
#' # Using default 'auto'/'sum_over_batch_size' reduction type.
#' bce <- loss_binary_crossentropy(from_logits = TRUE)
#' bce(y_true, y_pred)
#'
#' # Using 'sample_weight' attribute
#' bce(y_true, y_pred, sample_weight = c(0.8, 0.2))
#' # 0.243
#' # Using 'sum' reduction` type.
#' bce <- loss_binary_crossentropy(from_logits = TRUE, reduction = "sum")
#' bce(y_true, y_pred)
#'
#' # Using 'none' reduction type.
#' bce <- loss_binary_crossentropy(from_logits = TRUE, reduction = NULL)
#' bce(y_true, y_pred)
#' ```
#'
#' **Default Usage:** (set `from_logits=FALSE`)
#'
#' ```{r}
#' # Make the following updates to the above "Recommended Usage" section
#' # 1. Set `from_logits=FALSE`
#' loss_binary_crossentropy() # OR ...('from_logits=FALSE')
#' # 2. Update `y_pred` to use probabilities instead of logits
#' y_pred <- c(0.6, 0.3, 0.2, 0.8) # OR [[0.6, 0.3], [0.2, 0.8]]
#' ```
#'
#' @returns
#' Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param from_logits
#' Whether to interpret `y_pred` as a tensor of
#' [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
#' assume that `y_pred` is probabilities (i.e., values in `[0, 1)).`
#'
#' @param label_smoothing
#' Float in range `[0, 1].` When 0, no smoothing occurs.
#' When > 0, we compute the loss between the predicted labels
#' and a smoothed version of the true labels, where the smoothing
#' squeezes the labels towards 0.5. Larger values of
#' `label_smoothing` correspond to heavier smoothing.
#'
#' @param axis
#' The axis along which to compute crossentropy (the features axis).
#' Defaults to `-1`.
#'
#' @param reduction
#' Type of reduction to apply to the loss. In almost all cases
#' this should be `"sum_over_batch_size"`.
#' Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.
#'
#' @param name
#' Optional name for the loss instance.
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
#' @seealso
#' + <https://keras.io/api/losses/probabilistic_losses#binarycrossentropy-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy>
#'
#' @tether keras.losses.BinaryCrossentropy
loss_binary_crossentropy <-
function (y_true, y_pred, from_logits = FALSE, label_smoothing = 0,
    axis = -1L, ..., reduction = "sum_over_batch_size", name = "binary_crossentropy")
{
    args <- capture_args(list(axis = as_axis, y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x)))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$BinaryCrossentropy
    else keras$losses$binary_crossentropy
    do.call(callable, args)
}


#' Computes focal cross-entropy loss between true labels and predictions.
#'
#' @description
#' According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
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
#' Binary cross-entropy loss is often used for binary (0 or 1) classification
#' tasks. The loss function requires the following inputs:
#'
#' - `y_true` (true label): This is either 0 or 1.
#' - `y_pred` (predicted value): This is the model's prediction, i.e, a single
#'     floating-point value which either represents a
#'     [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in `[-inf, inf]`
#'     when `from_logits=TRUE`) or a probability (i.e, value in `[0., 1.]` when
#'     `from_logits=FALSE`).
#'
#' According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
#' helps to apply a "focal factor" to down-weight easy examples and focus more
#' on hard examples. By default, the focal tensor is computed as follows:
#'
#' `focal_factor = (1 - output) ** gamma` for class 1
#' `focal_factor = output ** gamma` for class 0
#' where `gamma` is a focusing parameter. When `gamma=0`, this function is
#' equivalent to the binary crossentropy loss.
#'
#' # Examples
#' ```{r}
#' y_true <- rbind(c(0, 1), c(0, 0))
#' y_pred <- rbind(c(0.6, 0.4), c(0.4, 0.6))
#' loss <- loss_binary_focal_crossentropy(y_true, y_pred, gamma = 2)
#' loss
#' ```
#' With the `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(
#'     loss = loss_binary_focal_crossentropy(
#'         gamma = 2.0, from_logits = TRUE),
#'     ...
#' )
#' ```
#'
#' As a standalone function:
#'
#' ```{r}
#' # Example 1: (batch_size = 1, number of samples = 4)
#' y_true <- op_array(c(0, 1, 0, 0))
#' y_pred <- op_array(c(-18.6, 0.51, 2.94, -12.8))
#' loss <- loss_binary_focal_crossentropy(gamma = 2, from_logits = TRUE)
#' loss(y_true, y_pred)
#' ```
#'
#' ```{r}
#' # Apply class weight
#' loss <- loss_binary_focal_crossentropy(
#'   apply_class_balancing = TRUE, gamma = 2, from_logits = TRUE)
#' loss(y_true, y_pred)
#' ```
#'
#' ```{r}
#' # Example 2: (batch_size = 2, number of samples = 4)
#' y_true <- rbind(c(0, 1), c(0, 0))
#' y_pred <- rbind(c(-18.6, 0.51), c(2.94, -12.8))
#' # Using default 'auto'/'sum_over_batch_size' reduction type.
#' loss <- loss_binary_focal_crossentropy(
#'     gamma = 3, from_logits = TRUE)
#' loss(y_true, y_pred)
#' ```
#'
#' ```{r}
#' # Apply class weight
#' loss <- loss_binary_focal_crossentropy(
#'      apply_class_balancing = TRUE, gamma = 3, from_logits = TRUE)
#' loss(y_true, y_pred)
#' ```
#'
#' ```{r}
#' # Using 'sample_weight' attribute with focal effect
#' loss <- loss_binary_focal_crossentropy(
#'     gamma = 3, from_logits = TRUE)
#' loss(y_true, y_pred, sample_weight = c(0.8, 0.2))
#' ```
#'
#' ```{r}
#' # Apply class weight
#' loss <- loss_binary_focal_crossentropy(
#'      apply_class_balancing = TRUE, gamma = 3, from_logits = TRUE)
#' loss(y_true, y_pred, sample_weight = c(0.8, 0.2))
#' ```
#'
#' ```{r}
#' # Using 'sum' reduction` type.
#' loss <- loss_binary_focal_crossentropy(
#'     gamma = 4, from_logits = TRUE,
#'     reduction = "sum")
#' loss(y_true, y_pred)
#' ```
#'
#' ```{r}
#' # Apply class weight
#' loss <- loss_binary_focal_crossentropy(
#'     apply_class_balancing = TRUE, gamma = 4, from_logits = TRUE,
#'     reduction = "sum")
#' loss(y_true, y_pred)
#' ```
#'
#' ```{r}
#' # Using 'none' reduction type.
#' loss <- loss_binary_focal_crossentropy(
#'     gamma = 5, from_logits = TRUE,
#'     reduction = NULL)
#' loss(y_true, y_pred)
#' ```
#'
#' ```{r}
#' # Apply class weight
#' loss <- loss_binary_focal_crossentropy(
#'     apply_class_balancing = TRUE, gamma = 5, from_logits = TRUE,
#'     reduction = NULL)
#' loss(y_true, y_pred)
#' ```
#'
#' @returns
#' Binary focal crossentropy loss value
#' with shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param apply_class_balancing
#' A bool, whether to apply weight balancing on the
#' binary classes 0 and 1.
#'
#' @param alpha
#' A weight balancing factor for class 1, default is `0.25` as
#' mentioned in reference [Lin et al., 2018](
#' https://arxiv.org/pdf/1708.02002.pdf).  The weight for class 0 is
#' `1.0 - alpha`.
#'
#' @param gamma
#' A focusing parameter used to compute the focal factor, default is
#' `2.0` as mentioned in the reference
#' [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf).
#'
#' @param from_logits
#' Whether to interpret `y_pred` as a tensor of
#' [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
#' assume that `y_pred` are probabilities (i.e., values in `[0, 1]`).
#'
#' @param label_smoothing
#' Float in `[0, 1]`. When `0`, no smoothing occurs.
#' When > `0`, we compute the loss between the predicted labels
#' and a smoothed version of the true labels, where the smoothing
#' squeezes the labels towards `0.5`.
#' Larger values of `label_smoothing` correspond to heavier smoothing.
#'
#' @param axis
#' The axis along which to compute crossentropy (the features axis).
#' Defaults to `-1`.
#'
#' @param reduction
#' Type of reduction to apply to the loss. In almost all cases
#' this should be `"sum_over_batch_size"`.
#' Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.
#'
#' @param name
#' Optional name for the loss instance.
#'
#' @param y_true
#' Ground truth values, of shape `(batch_size, d0, .. dN)`.
#'
#' @param y_pred
#' The predicted values, of shape `(batch_size, d0, .. dN)`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family losses
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryFocalCrossentropy>
#'
#' @tether keras.losses.BinaryFocalCrossentropy
loss_binary_focal_crossentropy <-
function (y_true, y_pred, apply_class_balancing = FALSE,
    alpha = 0.25, gamma = 2, from_logits = FALSE, label_smoothing = 0,
    axis = -1L, ..., reduction = "sum_over_batch_size", name = "binary_focal_crossentropy")
{
    args <- capture_args(list(axis = as_axis, y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x)))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$BinaryFocalCrossentropy
    else keras$losses$binary_focal_crossentropy
    do.call(callable, args)
}


#' Computes the crossentropy loss between the labels and predictions.
#'
#' @description
#' Use this crossentropy loss function when there are two or more label
#' classes. We expect labels to be provided in a `one_hot` representation. If
#' you want to provide labels as integers, please use
#' `SparseCategoricalCrossentropy` loss. There should be `num_classes` floating
#' point values per feature, i.e., the shape of both `y_pred` and `y_true` are
#' `[batch_size, num_classes]`.
#'
#' # Examples
#' ```{r}
#' y_true <- rbind(c(0, 1, 0), c(0, 0, 1))
#' y_pred <- rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1))
#' loss <- loss_categorical_crossentropy(y_true, y_pred)
#' loss
#' ```
#' Standalone usage:
#'
#' ```{r}
#' y_true <- rbind(c(0, 1, 0), c(0, 0, 1))
#' y_pred <- rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1))
#' # Using 'auto'/'sum_over_batch_size' reduction type.
#' cce <- loss_categorical_crossentropy()
#' cce(y_true, y_pred)
#' ```
#'
#' ```{r}
#' # Calling with 'sample_weight'.
#' cce(y_true, y_pred, sample_weight = op_array(c(0.3, 0.7)))
#' ```
#'
#' ```{r}
#' # Using 'sum' reduction type.
#' cce <- loss_categorical_crossentropy(reduction = "sum")
#' cce(y_true, y_pred)
#' ```
#'
#' ```{r}
#' # Using 'none' reduction type.
#' cce <- loss_categorical_crossentropy(reduction = NULL)
#' cce(y_true, y_pred)
#' ```
#'
#' Usage with the `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(optimizer = 'sgd',
#'               loss=loss_categorical_crossentropy())
#' ```
#'
#' @returns
#' Categorical crossentropy loss value.
#'
#' @param from_logits
#' Whether `y_pred` is expected to be a logits tensor. By
#' default, we assume that `y_pred` encodes a probability distribution.
#'
#' @param label_smoothing
#' Float in `[0, 1].` When > 0, label values are smoothed,
#' meaning the confidence on label values are relaxed. For example, if
#' `0.1`, use `0.1 / num_classes` for non-target labels and
#' `0.9 + 0.1 / num_classes` for target labels.
#'
#' @param axis
#' The axis along which to compute crossentropy (the features
#' axis). Defaults to `-1`.
#'
#' @param reduction
#' Type of reduction to apply to the loss. In almost all cases
#' this should be `"sum_over_batch_size"`.
#' Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.
#'
#' @param name
#' Optional name for the loss instance.
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
#' @export
#' @family losses
#' @seealso
#' + <https://keras.io/api/losses/probabilistic_losses#categoricalcrossentropy-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy>
#'
#' @tether keras.losses.CategoricalCrossentropy
loss_categorical_crossentropy <-
function (y_true, y_pred, from_logits = FALSE, label_smoothing = 0,
    axis = -1L, ..., reduction = "sum_over_batch_size", name = "categorical_crossentropy")
{
    args <- capture_args(list(axis = as_axis, y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x)))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$CategoricalCrossentropy
    else keras$losses$categorical_crossentropy
    do.call(callable, args)
}


#' Computes the alpha balanced focal crossentropy loss.
#'
#' @description
#' Use this crossentropy loss function when there are two or more label
#' classes and if you want to handle class imbalance without using
#' `class_weights`. We expect labels to be provided in a `one_hot`
#' representation.
#'
#' According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
#' helps to apply a focal factor to down-weight easy examples and focus more on
#' hard examples. The general formula for the focal loss (FL)
#' is as follows:
#'
#' `FL(p_t) = (1 - p_t)^gamma * log(p_t)`
#'
#' where `p_t` is defined as follows:
#' `p_t = output if y_true == 1, else 1 - output`
#'
#' `(1 - p_t)^gamma` is the `modulating_factor`, where `gamma` is a focusing
#' parameter. When `gamma` = 0, there is no focal effect on the cross entropy.
#' `gamma` reduces the importance given to simple examples in a smooth manner.
#'
#' The authors use alpha-balanced variant of focal loss (FL) in the paper:
#' `FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)`
#'
#' where `alpha` is the weight factor for the classes. If `alpha` = 1, the
#' loss won't be able to handle class imbalance properly as all
#' classes will have the same weight. This can be a constant or a list of
#' constants. If alpha is a list, it must have the same length as the number
#' of classes.
#'
#' The formula above can be generalized to:
#' `FL(p_t) = alpha * (1 - p_t)^gamma * CrossEntropy(y_true, y_pred)`
#'
#' where minus comes from `CrossEntropy(y_true, y_pred)` (CE).
#'
#' Extending this to multi-class case is straightforward:
#' `FL(p_t) = alpha * (1 - p_t) ** gamma * CategoricalCE(y_true, y_pred)`
#'
#' In the snippet below, there is `num_classes` floating pointing values per
#' example. The shape of both `y_pred` and `y_true` are
#' `(batch_size, num_classes)`.
#'
#' # Examples
#' ```{r}
#' y_true <- rbind(c(0, 1, 0), c(0, 0, 1))
#' y_pred <- rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1))
#' loss <- loss_categorical_focal_crossentropy(y_true, y_pred)
#' loss
#' ```
#' Standalone usage:
#'
#' ```{r}
#' y_true <- rbind(c(0, 1, 0), c(0, 0, 1))
#' y_pred <- rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1))
#' # Using 'auto'/'sum_over_batch_size' reduction type.
#' cce <- loss_categorical_focal_crossentropy()
#' cce(y_true, y_pred)
#' ```
#'
#' ```{r}
#' # Calling with 'sample_weight'.
#' cce(y_true, y_pred, sample_weight = op_array(c(0.3, 0.7)))
#' ```
#'
#' ```{r}
#' # Using 'sum' reduction type.
#' cce <- loss_categorical_focal_crossentropy(reduction = "sum")
#' cce(y_true, y_pred)
#' ```
#'
#' ```{r}
#' # Using 'none' reduction type.
#' cce <- loss_categorical_focal_crossentropy(reduction = NULL)
#' cce(y_true, y_pred)
#' ```
#'
#' Usage with the `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(
#'   optimizer = 'adam',
#'   loss = loss_categorical_focal_crossentropy())
#' ```
#'
#' @returns
#' Categorical focal crossentropy loss value.
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
#' Whether `output` is expected to be a logits tensor. By
#' default, we consider that `output` encodes a probability
#' distribution.
#'
#' @param label_smoothing
#' Float in `[0, 1].` When > 0, label values are smoothed,
#' meaning the confidence on label values are relaxed. For example, if
#' `0.1`, use `0.1 / num_classes` for non-target labels and
#' `0.9 + 0.1 / num_classes` for target labels.
#'
#' @param axis
#' The axis along which to compute crossentropy (the features
#' axis). Defaults to `-1`.
#'
#' @param reduction
#' Type of reduction to apply to the loss. In almost all cases
#' this should be `"sum_over_batch_size"`.
#' Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.
#'
#' @param name
#' Optional name for the loss instance.
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
#' @export
#' @family losses
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalFocalCrossentropy>
#'
#' @tether keras.losses.CategoricalFocalCrossentropy
loss_categorical_focal_crossentropy <-
function (y_true, y_pred, alpha = 0.25, gamma = 2,
    from_logits = FALSE, label_smoothing = 0, axis = -1L, ...,
    reduction = "sum_over_batch_size", name = "categorical_focal_crossentropy")
{
    args <- capture_args(list(axis = as_axis, y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x)))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$CategoricalFocalCrossentropy
    else keras$losses$categorical_focal_crossentropy
    do.call(callable, args)
}


#' Computes the categorical hinge loss between `y_true` & `y_pred`.
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
#' # Examples
#' ```{r}
#' y_true <- rbind(c(0, 1), c(0, 0))
#' y_pred <- rbind(c(0.6, 0.4), c(0.4, 0.6))
#' loss <- loss_categorical_hinge(y_true, y_pred)
#' ```
#'
#' @returns
#' Categorical hinge loss values with shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param reduction
#' Type of reduction to apply to the loss. In almost all cases
#' this should be `"sum_over_batch_size"`.
#' Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.
#'
#' @param name
#' Optional name for the loss instance.
#'
#' @param y_true
#' The ground truth values. `y_true` values are expected to be
#' either `{-1, +1}` or `{0, 1}` (i.e. a one-hot-encoded tensor) with
#' shape <- `[batch_size, d0, .. dN]`.
#'
#' @param y_pred
#' The predicted values with shape = `[batch_size, d0, .. dN]`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family losses
#' @seealso
#' + <https://keras.io/api/losses/hinge_losses#categoricalhinge-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalHinge>
#'
#' @tether keras.losses.CategoricalHinge
loss_categorical_hinge <-
function (y_true, y_pred, ..., reduction = "sum_over_batch_size",
    name = "categorical_hinge")
{
    args <- capture_args(list(y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x)))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$CategoricalHinge
    else keras$losses$categorical_hinge
    do.call(callable, args)
}


#' Computes the cosine similarity between `y_true` & `y_pred`.
#'
#' @description
#' Formula:
#' ```{r, eval = FALSE}
#' loss <- -sum(l2_norm(y_true) * l2_norm(y_pred))
#' ```
#'
#' Note that it is a number between -1 and 1. When it is a negative number
#' between -1 and 0, 0 indicates orthogonality and values closer to -1
#' indicate greater similarity. This makes it usable as a loss function in a
#' setting where you try to maximize the proximity between predictions and
#' targets. If either `y_true` or `y_pred` is a zero vector, cosine
#' similarity will be 0 regardless of the proximity between predictions
#' and targets.
#'
#' # Examples
#' ```{r}
#' y_true <- rbind(c(0., 1.), c(1., 1.), c(1., 1.))
#' y_pred <- rbind(c(1., 0.), c(1., 1.), c(-1., -1.))
#' loss <- loss_cosine_similarity(y_true, y_pred, axis=-1)
#' loss
#' ```
#'
#' @returns
#' Cosine similarity tensor.
#'
#' @param axis
#' The axis along which the cosine similarity is computed
#' (the features axis). Defaults to `-1`.
#'
#' @param reduction
#' Type of reduction to apply to the loss. In almost all cases
#' this should be `"sum_over_batch_size"`.
#' Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.
#'
#' @param name
#' Optional name for the loss instance.
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
#' @seealso
#' + <https://keras.io/api/losses/regression_losses#cosinesimilarity-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/CosineSimilarity>
#'
#' @tether keras.losses.CosineSimilarity
loss_cosine_similarity <-
function (y_true, y_pred, axis = -1L, ..., reduction = "sum_over_batch_size",
    name = "cosine_similarity")
{
    args <- capture_args(list(axis = as_axis, y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x)))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$CosineSimilarity
    else keras$losses$cosine_similarity
    do.call(callable, args)
}


#' Computes the hinge loss between `y_true` & `y_pred`.
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
#' # Examples
#' ```{r}
#' y_true <- array(sample(c(-1,1), 6, replace = TRUE), dim = c(2, 3))
#' y_pred <- random_uniform(c(2, 3))
#' loss <- loss_hinge(y_true, y_pred)
#' loss
#' ```
#'
#' @returns
#' Hinge loss values with shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param reduction
#' Type of reduction to apply to the loss. In almost all cases
#' this should be `"sum_over_batch_size"`.
#' Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.
#'
#' @param name
#' Optional name for the loss instance.
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
#' @export
#' @family losses
#' @seealso
#' + <https://keras.io/api/losses/hinge_losses#hinge-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/Hinge>
#'
#' @tether keras.losses.Hinge
loss_hinge <-
function (y_true, y_pred, ..., reduction = "sum_over_batch_size",
    name = "hinge")
{
    args <- capture_args(list(y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x)))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$Hinge
    else keras$losses$hinge
    do.call(callable, args)
}


#' Computes the Huber loss between `y_true` & `y_pred`.
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
#' @param delta
#' A float, the point where the Huber loss function changes from a
#' quadratic to linear. Defaults to `1.0`.
#'
#' @param reduction
#' Type of reduction to apply to loss. Options are `"sum"`,
#' `"sum_over_batch_size"` or `NULL`. Defaults to
#' `"sum_over_batch_size"`.
#'
#' @param name
#' Optional name for the instance.
#'
#' @param y_true
#' tensor of true targets.
#'
#' @param y_pred
#' tensor of predicted targets.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family losses
#' @seealso
#' + <https://keras.io/api/losses/regression_losses#huber-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/Huber>
#'
#' @tether keras.losses.Huber
loss_huber <-
function (y_true, y_pred, delta = 1, ..., reduction = "sum_over_batch_size",
    name = "huber_loss")
{
    args <- capture_args(list(y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x)))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$Huber
    else keras$losses$huber
    do.call(callable, args)
}


#' Computes Kullback-Leibler divergence loss between `y_true` & `y_pred`.
#'
#' @description
#' Formula:
#'
#' ```{r, eval=FALSE}
#' loss <- y_true * log(y_true / y_pred)
#' ```
#'
#' # Examples
#' ```{r}
#' y_true <- random_uniform(c(2, 3), 0, 2)
#' y_pred <- random_uniform(c(2,3))
#' loss <- loss_kl_divergence(y_true, y_pred)
#' loss
#' ```
#'
#' @returns
#' KL Divergence loss values with shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param reduction
#' Type of reduction to apply to the loss. In almost all cases
#' this should be `"sum_over_batch_size"`.
#' Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.
#'
#' @param name
#' Optional name for the loss instance.
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
#' @seealso
#' + <https://keras.io/api/losses/probabilistic_losses#kldivergence-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/KLDivergence>
#'
#' @tether keras.losses.KLDivergence
loss_kl_divergence <-
function (y_true, y_pred, ..., reduction = "sum_over_batch_size",
    name = "kl_divergence")
{
    args <- capture_args(list(y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x)))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$KLDivergence
    else keras$losses$kl_divergence
    do.call(callable, args)
}


#' Computes the logarithm of the hyperbolic cosine of the prediction error.
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
#' loss <- loss_log_cosh(y_true, y_pred)
#' # 0.108
#' ```
#'
#' @returns
#' Logcosh error values with shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param reduction
#' Type of reduction to apply to loss. Options are `"sum"`,
#' `"sum_over_batch_size"` or `NULL`. Defaults to
#' `"sum_over_batch_size"`.
#'
#' @param name
#' Optional name for the instance.
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
#' @family losses
#' @seealso
#' + <https://keras.io/api/losses/regression_losses#logcosh-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/LogCosh>
#'
#' @tether keras.losses.LogCosh
loss_log_cosh <-
function (y_true, y_pred, ..., reduction = "sum_over_batch_size",
    name = "log_cosh")
{
    args <- capture_args(list(y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x)))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$LogCosh
    else keras$losses$log_cosh
    do.call(callable, args)
}


#' Computes the mean of absolute difference between labels and predictions.
#'
#' @description
#' Formula:
#'
#' ```{r, eval = FALSE}
#' loss <- mean(abs(y_true - y_pred))
#' ```
#'
#' # Examples
#' ```{r}
#' y_true <- random_uniform(c(2, 3), 0, 2)
#' y_pred <- random_uniform(c(2, 3))
#' loss <- loss_mean_absolute_error(y_true, y_pred)
#' ```
#'
#' @returns
#' Mean absolute error values with shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param reduction
#' Type of reduction to apply to the loss. In almost all cases
#' this should be `"sum_over_batch_size"`.
#' Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.
#'
#' @param name
#' Optional name for the loss instance.
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
#' @family losses
#' @seealso
#' + <https://keras.io/api/losses/regression_losses#meanabsoluteerror-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanAbsoluteError>
#'
#' @tether keras.losses.MeanAbsoluteError
loss_mean_absolute_error <-
function (y_true, y_pred, ..., reduction = "sum_over_batch_size",
    name = "mean_absolute_error")
{
    args <- capture_args(list(y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x)))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$MeanAbsoluteError
    else keras$losses$mean_absolute_error
    do.call(callable, args)
}


#' Computes the mean absolute percentage error between `y_true` and `y_pred`.
#'
#' @description
#' Formula:
#'
#' ```{r, eval = FALSE}
#' loss <- 100 * op_mean(op_abs((y_true - y_pred) / y_true),
#'                       axis=-1)
#' ```
#'
#' Division by zero is prevented by dividing by `max(y_true, epsilon)`
#' where `epsilon = config_epsilon()`
#' (default to `1e-7`).
#'
#' # Examples
#' ```{r}
#' y_true <- random_uniform(c(2, 3))
#' y_pred <- random_uniform(c(2, 3))
#' loss <- loss_mean_absolute_percentage_error(y_true, y_pred)
#' ```
#'
#' @returns
#' Mean absolute percentage error values with shape = `[batch_size, d0, ..dN-1]`.
#'
#' @param reduction
#' Type of reduction to apply to the loss. In almost all cases
#' this should be `"sum_over_batch_size"`.
#' Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.
#'
#' @param name
#' Optional name for the loss instance.
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
#' @family losses
#' @seealso
#' + <https://keras.io/api/losses/regression_losses#meanabsolutepercentageerror-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanAbsolutePercentageError>
#'
#' @tether keras.losses.MeanAbsolutePercentageError
loss_mean_absolute_percentage_error <-
function (y_true, y_pred, ..., reduction = "sum_over_batch_size",
    name = "mean_absolute_percentage_error")
{
    args <- capture_args(list(y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x)))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$MeanAbsolutePercentageError
    else keras$losses$mean_absolute_percentage_error
    do.call(callable, args)
}


#' Computes the mean of squares of errors between labels and predictions.
#'
#' @description
#' Formula:
#'
#' ```{r, eval=FALSE}
#' loss <- mean(square(y_true - y_pred))
#' ```
#'
#' # Examples
#' ```{r}
#' y_true <- random_uniform(c(2, 3), 0, 2)
#' y_pred <- random_uniform(c(2, 3))
#' loss <- loss_mean_squared_error(y_true, y_pred)
#' ```
#'
#' @returns
#' Mean squared error values with shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param reduction
#' Type of reduction to apply to the loss. In almost all cases
#' this should be `"sum_over_batch_size"`.
#' Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.
#'
#' @param name
#' Optional name for the loss instance.
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
#' @family losses
#' @seealso
#' + <https://keras.io/api/losses/regression_losses#meansquarederror-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError>
#'
#' @tether keras.losses.MeanSquaredError
loss_mean_squared_error <-
function (y_true, y_pred, ..., reduction = "sum_over_batch_size",
    name = "mean_squared_error")
{
    args <- capture_args(list(y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x)))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$MeanSquaredError
    else keras$losses$mean_squared_error
    do.call(callable, args)
}


#' Computes the mean squared logarithmic error between `y_true` and `y_pred`.
#'
#' @description
#' Note that `y_pred` and `y_true` cannot be less or equal to `0`. Negative
#' values and `0` values will be replaced with `config_epsilon()`
#' (default to `1e-7`).
#'
#' Formula:
#'
#' ```{r, eval = FALSE}
#' loss <- mean(square(log(y_true + 1) - log(y_pred + 1)))
#' ```
#'
#' # Examples
#' ```{r}
#' y_true <- random_uniform(c(2, 3), 0, 2)
#' y_pred <- random_uniform(c(2, 3))
#' loss <- loss_mean_squared_logarithmic_error(y_true, y_pred)
#' ```
#'
#' @returns
#' Mean squared logarithmic error values with shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param reduction
#' Type of reduction to apply to the loss. In almost all cases
#' this should be `"sum_over_batch_size"`.
#' Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.
#'
#' @param name
#' Optional name for the loss instance.
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
#' @family losses
#' @seealso
#' + <https://keras.io/api/losses/regression_losses#meansquaredlogarithmicerror-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredLogarithmicError>
#'
#' @tether keras.losses.MeanSquaredLogarithmicError
loss_mean_squared_logarithmic_error <-
function (y_true, y_pred, ..., reduction = "sum_over_batch_size",
    name = "mean_squared_logarithmic_error")
{
    args <- capture_args(list(y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x)))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$MeanSquaredLogarithmicError
    else keras$losses$mean_squared_logarithmic_error
    do.call(callable, args)
}


#' Computes the Poisson loss between `y_true` & `y_pred`.
#'
#' @description
#' Formula:
#'
#' ```{r, eval=FALSE}
#' loss <- y_pred - y_true * log(y_pred)
#' ```
#'
#' # Examples
#' ```{r}
#' y_true <- random_uniform(c(2, 3), 0, 2)
#' y_pred <- random_uniform(c(2, 3))
#' loss <- loss_poisson(y_true, y_pred)
#' loss
#' ```
#'
#' @returns
#' Poisson loss values with shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param reduction
#' Type of reduction to apply to the loss. In almost all cases
#' this should be `"sum_over_batch_size"`.
#' Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.
#'
#' @param name
#' Optional name for the loss instance.
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
#' @seealso
#' + <https://keras.io/api/losses/probabilistic_losses#poisson-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/Poisson>
#'
#' @tether keras.losses.Poisson
loss_poisson <-
function (y_true, y_pred, ..., reduction = "sum_over_batch_size",
    name = "poisson")
{
    args <- capture_args(list(y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x)))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$Poisson
    else keras$losses$poisson
    do.call(callable, args)
}


#' Computes the crossentropy loss between the labels and predictions.
#'
#' @description
#' Use this crossentropy loss function when there are two or more label
#' classes.  We expect labels to be provided as integers. If you want to
#' provide labels using `one-hot` representation, please use
#' `CategoricalCrossentropy` loss.  There should be `# classes` floating point
#' values per feature for `y_pred` and a single floating point value per
#' feature for `y_true`.
#'
#' In the snippet below, there is a single floating point value per example for
#' `y_true` and `num_classes` floating pointing values per example for
#' `y_pred`. The shape of `y_true` is `[batch_size]` and the shape of `y_pred`
#' is `[batch_size, num_classes]`.
#'
#' # Examples
#' ```{r}
#' y_true <- c(1, 2)
#' y_pred <- rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1))
#' loss <- loss_sparse_categorical_crossentropy(y_true, y_pred)
#' loss
#' ```
#' ```{r}
#' y_true <- c(1, 2)
#' y_pred <- rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1))
#' # Using 'auto'/'sum_over_batch_size' reduction type.
#' scce <- loss_sparse_categorical_crossentropy()
#' scce(op_array(y_true), op_array(y_pred))
#' # 1.177
#' ```
#'
#' ```{r}
#' # Calling with 'sample_weight'.
#' scce(op_array(y_true), op_array(y_pred), sample_weight = op_array(c(0.3, 0.7)))
#' ```
#'
#' ```{r}
#' # Using 'sum' reduction type.
#' scce <- loss_sparse_categorical_crossentropy(reduction="sum")
#' scce(op_array(y_true), op_array(y_pred))
#' # 2.354
#' ```
#'
#' ```{r}
#' # Using 'none' reduction type.
#' scce <- loss_sparse_categorical_crossentropy(reduction=NULL)
#' scce(op_array(y_true), op_array(y_pred))
#' # array([0.0513, 2.303], dtype=float32)
#' ```
#'
#' Usage with the `compile()` API:
#'
#' ```{r, eval = FALSE}
#' model %>% compile(optimizer = 'sgd',
#'                   loss = loss_sparse_categorical_crossentropy())
#' ```
#'
#' @returns
#' Sparse categorical crossentropy loss value.
#'
#' @param from_logits
#' Whether `y_pred` is expected to be a logits tensor. By
#' default, we assume that `y_pred` encodes a probability distribution.
#'
#' @param reduction
#' Type of reduction to apply to the loss. In almost all cases
#' this should be `"sum_over_batch_size"`.
#' Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.
#'
#' @param name
#' Optional name for the loss instance.
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
#' @param axis
#' Defaults to `-1`. The dimension along which the entropy is
#' computed.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family losses
#' @seealso
#' + <https://keras.io/api/losses/probabilistic_losses#sparsecategoricalcrossentropy-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy>
#'
#' @tether keras.losses.SparseCategoricalCrossentropy
loss_sparse_categorical_crossentropy <-
function (y_true, y_pred, from_logits = FALSE, ignore_class = NULL,
    axis = -1L, ..., reduction = "sum_over_batch_size", name = "sparse_categorical_crossentropy")
{
    args <- capture_args(list(ignore_class = as_integer, y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), axis = as_axis))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$SparseCategoricalCrossentropy
    else keras$losses$sparse_categorical_crossentropy
    do.call(callable, args)
}


#' Computes the squared hinge loss between `y_true` & `y_pred`.
#'
#' @description
#' Formula:
#'
#' ```{r, eval=FALSE}
#' loss <- square(maximum(1 - y_true * y_pred, 0))
#' ```
#'
#' `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
#' provided we will convert them to -1 or 1.
#'
#' # Examples
#' ```{r}
#' y_true <- array(sample(c(-1,1), 6, replace = TRUE), dim = c(2, 3))
#' y_pred <- random_uniform(c(2, 3))
#' loss <- loss_squared_hinge(y_true, y_pred)
#' ```
#'
#' @returns
#' Squared hinge loss values with shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param reduction
#' Type of reduction to apply to the loss. In almost all cases
#' this should be `"sum_over_batch_size"`.
#' Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.
#'
#' @param name
#' Optional name for the loss instance.
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
#' @export
#' @family losses
#' @seealso
#' + <https://keras.io/api/losses/hinge_losses#squaredhinge-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/SquaredHinge>
#'
#' @tether keras.losses.SquaredHinge
loss_squared_hinge <-
function (y_true, y_pred, ..., reduction = "sum_over_batch_size",
    name = "squared_hinge")
{
    args <- capture_args(list(y_true = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x), y_pred = function (x)
    if (inherits(x, "python.builtin.object"))
        x
    else np_array(x)))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$SquaredHinge
    else keras$losses$squared_hinge
    do.call(callable, args)
}


#' CTC (Connectionist Temporal Classification) loss.
#'
#' @param y_true
#' A tensor of shape `(batch_size, target_max_length)` containing
#' the true labels in integer format. `0` always represents
#' the blank/mask index and should not be used for classes.
#'
#' @param y_pred
#' A tensor of shape `(batch_size, output_max_length, num_classes)`
#' containing logits (the output of your model).
#' They should *not* be normalized via softmax.
#'
#' @param name
#' String, name for the object
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @tether keras.losses.CTC
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/CTC>
loss_ctc <-
function (y_true, y_pred, ..., reduction = "sum_over_batch_size",
    name = "sparse_categorical_crossentropy")
{
    args <- capture_args(list(y_true = as_py_array, y_pred = as_py_array))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$CTC
    else keras$losses$ctc
    do.call(callable, args)
}




#' @importFrom reticulate py_to_r_wrapper
#' @export
#' @keywords internal
#' Wrapper for Loss/Metric instances that automatically coerces `y_true` and `y_pred` to the appropriate type.
py_to_r_wrapper.keras.src.losses.loss.Loss <- function(x) {
  force(x)
  as.function.default(c(formals(x), quote({
    args <- capture_args(list(y_true = as_py_array,
                               y_pred = as_py_array,
                               sample_weight = as_py_array))
    do.call(x, args)
  })))
}
