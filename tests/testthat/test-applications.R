
context("applications")



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

    if (is_keras_available("2.2.0"))
      application_mobilenet_v2()
  }

  application_resnet50()
  application_vgg16()
  application_vgg19()
  application_inception_v3()
})

test_succeeds("can use any input_shape", {
  x <- application_vgg16(weights = NULL, input_shape = shape(NULL, NULL, 3), include_top = FALSE)
  expect_s3_class(x, "keras.engine.training.Model")
})
