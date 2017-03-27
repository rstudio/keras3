context("regularizers")

source("utils.R")

test_call_succeeds("regularizer_l1", {
  regularizer_l1()
})

test_call_succeeds("regularizer_l1l2", {
  regularizer_l1l2()
})

test_call_succeeds("regularizer_l2", {
  regularizer_l2()
})


