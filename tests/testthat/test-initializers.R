context("initializers")

source("utils.R")

test_initializer <- function(name) {
  initializer_fn <- eval(parse(text = paste0("initializer_", name)))
  test_call_succeeds(name, {
    model_sequential() %>% 
      layer_dense(32, input_shape = c(32,32), kernel_initializer = initializer_fn()) %>% 
      compile( 
        optimizer = 'sgd',
        loss='binary_crossentropy', 
        metrics='accuracy'
      )
  }) 
}


test_initializer("zeros")
test_initializer("ones")
test_initializer("constant")
test_initializer("random_normal")
test_initializer("random_uniform")
test_initializer("truncated_normal")
test_initializer("variance_scaling")
test_initializer("orthogonal")
test_initializer("identity")
test_initializer("glorot_normal")
test_initializer("glorot_uniform")
test_initializer("he_uniform")
test_initializer("he_normal")
test_initializer("lecun_uniform")


