
context("applications")

source("utils.R")


test_succeeds("keras pre-built models can be instantiated", {
  application_xception()
  application_resnet50()
  application_vgg16()
  application_vgg19()
  application_inception_v3()
})



