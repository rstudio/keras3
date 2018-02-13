
context("applications")

source("utils.R")


test_succeeds("keras pre-built models can be instantiated", {
  
  skip <- is.na(Sys.getenv("KERAS_TEST_APPLICATIONS", unset = NA)) && 
          is.na(Sys.getenv("KERAS_TEST_ALL", unset = NA))
  if (skip)
    return()
  
  if (is_backend("tensorflow")) {
    application_xception()
    if (is_keras_available("2.0.6"))
      application_mobilenet()
    
    if (is_keras_available("2.1.3")) {
      application_densenet121()
      application_densenet169()
      application_densenet201()
      
      application_nasnetlarge()
      application_nasnetmobile()
    }
  }
  
  application_resnet50()
  application_vgg16()
  application_vgg19()
  application_inception_v3()
})



