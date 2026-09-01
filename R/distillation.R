#' Feature distillation loss
#'
#' Feature distillation transfers knowledge from intermediate layers of the
#' teacher model to corresponding layers of the student model. This can help
#' the student learn better internal representations and often leads to better
#' performance than logits-only distillation. If layer names are omitted, the
#' final model outputs are used.
#'
#' # Examples
#'
#' ```{r}
#' loss <- distillation_feature(
#'   loss = "mse",
#'   teacher_layer_name = "teacher_features",
#'   student_layer_name = "student_features"
#' )
#' ```
#'
#' @param loss Loss function used for feature distillation. This can be a
#'   string identifier such as `"mse"`, `"cosine_similarity"`, or `"mae"`; a
#'   Keras loss instance; or a nested list of losses matching the layer-output
#'   structure. Use `NULL` within a list to skip distillation for that output.
#'   At least one loss must be non-`NULL`. Defaults to `"mse"`.
#' @param teacher_layer_name Name of the teacher layer from which to extract
#'   features. The final output is used when `NULL`, the default.
#' @param student_layer_name Name of the student layer from which to extract
#'   features. The final output is used when `NULL`, the default.
#'
#' @returns A `FeatureDistillation` instance.
#' @export
#' @family distillation
#' @tether keras.distillation.FeatureDistillation
distillation_feature <-
function(loss = "mse", teacher_layer_name = NULL, student_layer_name = NULL)
{
  args <- capture_args()
  do.call(keras$distillation$FeatureDistillation, args)
}


#' Logits distillation loss
#'
#' Transfers knowledge from final model outputs. This loss applies temperature
#' scaling to the teacher logits before computing the loss between teacher and
#' student predictions. It is the most common approach to knowledge
#' distillation.
#'
#' # Examples
#'
#' ```{r}
#' loss <- distillation_logits(temperature = 3)
#' ```
#'
#' @param temperature Temperature used for softmax scaling. Higher values
#'   produce softer probability distributions that are easier for the student
#'   to learn. Typical values range from 3 to 5. Defaults to 3.
#' @param loss Loss function used for distillation. This can be a string
#'   identifier such as `"kl_divergence"` or
#'   `"categorical_crossentropy"`; a Keras loss instance; or a nested list of
#'   losses matching the model-output structure. Use `NULL` within a list to
#'   skip distillation for that output. At least one loss must be non-`NULL`.
#'   Defaults to `"kl_divergence"`.
#'
#' @returns A `LogitsDistillation` instance.
#' @export
#' @family distillation
#' @tether keras.distillation.LogitsDistillation
distillation_logits <-
function(temperature = 3, loss = "kl_divergence")
{
  args <- capture_args()
  do.call(keras$distillation$LogitsDistillation, args)
}


#' Model for transferring knowledge from a teacher to a student
#'
#' A distiller trains a student model from both ground-truth labels and the
#' predictions or intermediate features of a frozen teacher model. After
#' training, access `model$student` to use the trained student independently.
#'
#' # Examples
#'
#' ```{r, eval = FALSE}
#' teacher <- keras_model_sequential(input_shape = 4) |>
#'   layer_dense(8, activation = "relu") |>
#'   layer_dense(3)
#' student <- keras_model_sequential(input_shape = 4) |>
#'   layer_dense(3)
#'
#' model <- distiller(
#'   teacher = teacher,
#'   student = student,
#'   distillation_losses = distillation_logits(temperature = 3)
#' )
#' model |> compile(optimizer = "adam", loss = "mse")
#' ```
#'
#' @param teacher Trained Keras model that provides the knowledge to transfer.
#'   The teacher is frozen by the distiller.
#' @param student Keras model to train.
#' @param distillation_losses A distillation loss or list of distillation
#'   losses, such as [`distillation_logits()`],
#'   [`distillation_feature()`], or compatible upstream distillation losses.
#' @param distillation_loss_weights Numeric vector of weights for the
#'   distillation losses. It must have the same length as
#'   `distillation_losses`. If `NULL`, equal weights are used.
#' @param student_loss_weight Weight of the student's supervised loss. Must be
#'   between 0 and 1. Defaults to 0.5.
#' @param name Name of the distiller model. Defaults to `"distiller"`.
#' @param ... Additional arguments passed to the parent Keras `Model` class.
#'
#' @returns A Keras `Distiller` model.
#' @export
#' @family distillation
#' @family model creation
#' @tether keras.distillation.Distiller
distiller <-
function(teacher, student, distillation_losses,
         distillation_loss_weights = NULL, student_loss_weight = 0.5,
         name = "distiller", ...)
{
  args <- capture_args()
  do.call(keras$distillation$Distiller, args)
}
