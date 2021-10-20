
context("applications")



test_succeeds("keras pre-built models can be instantiated", {

  skip_applications <-
    is.na(Sys.getenv("KERAS_TEST_APPLICATIONS", unset = NA)) &&
    is.na(Sys.getenv("KERAS_TEST_ALL", unset = NA))

  if (skip_applications)
    testthat::skip("Skipping testing applications")

  expect_model <-   function(x) expect_s3_class(x, "keras.engine.training.Model")

  expect_model(application_xception() )
  expect_model(application_mobilenet())

  expect_model(application_densenet121())
  expect_model(application_densenet169())
  expect_model(application_densenet201())

  expect_model(application_nasnetlarge())
  expect_model(application_nasnetmobile())

  expect_model(application_resnet50())
  expect_model(application_vgg16())
  expect_model(application_vgg19())
  expect_model(application_inception_v3())

  expect_model(application_resnet50())
  expect_model(application_resnet101())
  expect_model(application_resnet152())
  expect_model(application_resnet50_v2())
  expect_model(application_resnet101_v2())
  expect_model(application_resnet152_v2())

  if (is_keras_available("2.2.0"))
    expect_model(application_mobilenet_v2())

  if(tf_version() >= "2.3") {
    expect_model(application_efficientnet_b0())
    expect_model(application_efficientnet_b1())
    expect_model(application_efficientnet_b2())
    expect_model(application_efficientnet_b3())
    expect_model(application_efficientnet_b4())
    expect_model(application_efficientnet_b5())
    expect_model(application_efficientnet_b6())
    expect_model(application_efficientnet_b7())
  }

  if(tf_version() >= "2.4") {
    expect_model(application_mobilenet_v3_large())
    expect_model(application_mobilenet_v3_small())
  }



  # can use any input_shape
  expect_model(application_vgg16(weights = NULL,
                                 input_shape = shape(NULL, NULL, 3),
                                 include_top = FALSE))

})
