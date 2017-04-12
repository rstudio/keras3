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
