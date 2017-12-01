context("utils")

source("utils.R")

test_call_succeeds("to_categorical", {
  runif(1000, min = 0, max = 9) %>% 
    round() %>%
    matrix(nrow = 1000, ncol = 1) %>% 
    to_categorical(num_classes = 10)
})


test_call_succeeds("get_file", {
  get_file("2010zipcode.zip", 
           origin = "https://www.irs.gov/pub/irs-soi/2010zipcode.zip", 
           cache_subdir = "tests")
})


test_call_succeeds("hdf5_matrix", {
  
  if (!keras:::have_h5py())
    skip("h5py not available for testing")
  
  X_train = hdf5_matrix('test.h5', 'my_data', start=0, end=150)
  y_train = hdf5_matrix('test.h5', 'my_labels', start=0, end=150)
})


test_call_succeeds("normalize", {
  data <- runif(1000, min = 0, max = 9) %>%  round() %>% matrix(nrow = 1000, ncol = 1)
  normalize(data)
})


test_call_succeeds("with_custom_object_scope", {
  
  if (!keras:::have_h5py())
    skip("h5py not available for testing")
  

  sparse_top_k_cat_acc <- function(y_pred, y_true){
    metric_sparse_top_k_categorical_accuracy(y_pred, y_true, k = 5)
  }
  
  with_custom_object_scope(c(top_k_acc = sparse_top_k_cat_acc), {
    
    model <- define_model()
    
    model %>% compile(
      loss = "binary_crossentropy",
      optimizer = optimizer_nadam(),
      metrics = "top_k_acc"
    )
    
    tmp <- tempfile("model", fileext = ".hdf5")
    save_model_hdf5(model, tmp)
    model <- load_model_hdf5(tmp)
     
    # generate dummy training data
    data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
    labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)
     
    model %>% fit(data, labels, epochs = 2)
    
  })
  
    
  
  

  
  
})
