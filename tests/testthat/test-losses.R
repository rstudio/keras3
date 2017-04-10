context("losses")

source("utils.R")

test_loss <- function(name) {
  loss_fn <- eval(parse(text = paste0("loss_", name)))
  test_call_succeeds(name, {
    keras_model_sequential() %>% 
      layer_dense(32, input_shape = shape(784)) %>% 
      compile( 
        optimizer = optimizer_sgd(),
        loss = loss_fn(), 
        metrics='accuracy'
      )
  }) 
}


test_loss("mean_squared_error")
test_loss("mean_absolute_error")
test_loss("mean_absolute_percentage_error")
test_loss("mean_squared_logarithmic_error")
test_loss("squared_hinge")
test_loss("hinge")
test_loss("categorical_crossentropy")
test_loss("sparse_categorical_crossentropy")
test_loss("binary_crossentropy")
test_loss("kullback_leibler_divergence")
test_loss("poisson")
test_loss("cosine_proximity")



