
context("applications")

source("utils.R")


test_succeeds("keras pre-built models can be instantiated", {
  
  skip <- is.na(Sys.getenv("KERAS_TEST_APPLICATIONS", unset = NA))
  if (skip)
    return()
  
  application_xception()
  application_resnet50()
  application_vgg16()
  application_vgg19()
  application_inception_v3()
})



