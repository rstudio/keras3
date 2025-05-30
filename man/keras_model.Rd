% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/model-creation.R
\name{keras_model}
\alias{keras_model}
\title{Keras Model (Functional API)}
\usage{
keras_model(inputs = NULL, outputs = NULL, ...)
}
\arguments{
\item{inputs}{Input tensor(s) (from \code{\link[=keras_input]{keras_input()}})}

\item{outputs}{Output tensors (from calling layers with \code{inputs})}

\item{...}{Any additional arguments}
}
\value{
A \code{Model} instance.
}
\description{
A model is a directed acyclic graph of layers.
}
\section{Examples}{
\if{html}{\out{<div class="sourceCode r">}}\preformatted{library(keras3)

# input tensor
inputs <- keras_input(shape = c(784))

# outputs compose input + dense layers
predictions <- inputs |>
  layer_dense(units = 64, activation = 'relu') |>
  layer_dense(units = 64, activation = 'relu') |>
  layer_dense(units = 10, activation = 'softmax')

# create and compile model
model <- keras_model(inputs = inputs, outputs = predictions)
model |> compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)
}\if{html}{\out{</div>}}
}

\seealso{
Other model functions: \cr
\code{\link{get_config}()} \cr
\code{\link{get_layer}()} \cr
\code{\link{get_state_tree}()} \cr
\code{\link{keras_model_sequential}()} \cr
\code{\link{pop_layer}()} \cr
\code{\link{set_state_tree}()} \cr
\code{\link{summary.keras.src.models.model.Model}()} \cr

Other model creation: \cr
\code{\link{keras_input}()} \cr
\code{\link{keras_model_sequential}()} \cr
}
\concept{model creation}
\concept{model functions}
