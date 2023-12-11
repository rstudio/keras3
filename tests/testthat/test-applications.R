
context("applications")

skip_applications <-
  is.na(Sys.getenv("KERAS_TEST_APPLICATIONS", unset = NA)) &&
  is.na(Sys.getenv("KERAS_TEST_ALL", unset = NA))

if (skip_applications)
  testthat::skip("Skipping testing applications")


test_succeeds("keras pre-built models can be instantiated", {

  expect_model <-   function(x) expect_s3_class(x, "keras.models.model.Model")

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

test_that("pre processing and decoding", {

  # test that the preprocessing and decoding functions work
  # for all applications
  applications <- c("application_convnext_base", "application_convnext_large",
    "application_convnext_small", "application_convnext_tiny", "application_convnext_xlarge",
    "application_densenet121", "application_densenet169", "application_densenet201",
    "application_efficientnet_b0", "application_efficientnet_b1",
    "application_efficientnet_b2", "application_efficientnet_b3",
    "application_efficientnet_b4", "application_efficientnet_b5",
    "application_efficientnet_b6", "application_efficientnet_b7",
    "application_efficientnet_v2b0", "application_efficientnet_v2b1",
    "application_efficientnet_v2b2", "application_efficientnet_v2b3",
    "application_efficientnet_v2l", "application_efficientnet_v2m",
    "application_efficientnet_v2s", "application_inception_resnet_v2",
    "application_inception_v3", "application_mobilenet", "application_mobilenet_v2",
    "application_mobilenet_v3_large", "application_mobilenet_v3_small",
    "application_nasnetlarge", "application_nasnetmobile", "application_resnet101",
    "application_resnet101_v2", "application_resnet152", "application_resnet152_v2",
    "application_resnet50", "application_resnet50_v2", "application_vgg16",
    "application_vgg19", "application_xception")

  for (application in applications) {
    model <- do.call(application, list(weights = NULL))
    expect_length(
      application_decode_predictions(model, matrix(runif(1000), ncol = 1000)),
      1
    )

    x <- random_normal(c(1, 224, 244, 3))
    expect_error(
      application_preprocess_inputs(model, x),
      regexp = NA
    )

  }

})

