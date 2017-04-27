
context("applications")

source("utils.R")


test_succeeds("keras pre-built models can be instantiated", {
  keras_model_xception()
  keras_model_resnet50()
  keras_model_vgg16()
  keras_model_vgg19()
  keras_model_inception_v3()
})



