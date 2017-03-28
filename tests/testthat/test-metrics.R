context("metrics")

source("utils.R")

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
    )
})