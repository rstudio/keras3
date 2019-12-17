context("Custom layers")

source("utils.R")

test_succeeds("Can create and use a custom layer", {
  
  layer_multiply_by_x <- Layer(
    classname = "MultiplyByX",
    
    initialize = function(x) {
      super()$`__init__`()
      self$x <- tensorflow::tf$constant(x)
    },
      
    call =  function(inputs, ...) {
      inputs * self$x
    }
    
  )
  
  layer_multiply_by_2 <- layer_multiply_by_x(x = 2)
  
  input <- layer_input(shape = 1)
  output <- layer_multiply_by_2(input)
  
  model <- keras_model(input, output)
  
  out <- predict(model, c(1,2,3,4,5))
  
  expect_equal(out, matrix(1:5, ncol = 1)*2)
  expect_equal(model$get_config()$layers[[2]]$class_name, "MultiplyByX")
})

test_succeeds("Can use custom layers in sequential models", {
  
  layer_multiply_by_x <- Layer(
    classname = "MultiplyByX",
    
    initialize = function(x) {
      super()$`__init__`()
      self$x <- tensorflow::tf$constant(x)
    },
    
    call =  function(inputs, ...) {
      inputs * self$x
    }
    
  )
  
  model <- keras_model_sequential() %>% 
    layer_multiply_by_x(2) %>% 
    layer_multiply_by_x(2)
  
  out <- predict(model, c(1,2,3,4,5))
  
  expect_equal(out, matrix(1:5, ncol = 1)*2*2)
})



