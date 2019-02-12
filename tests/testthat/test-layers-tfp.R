
# These tests correspond to (possibly shortened and/or modified versions of) the "docstring" tests in
#
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/layers/distribution_layer_test.py
#
# We don't test for correctness of the implementations as we're just consumers.
# Instead, we test that we can successfully use these layers in Keras models from R.


context("tensorflow probability layer methods")

source("utils.R")

load_tfp <- function() {
  tfp <- NULL
  makeActiveBinding("tfp", env = .GlobalEnv, function() {
    if (!is.null(tfp)) {
      return(tfp)
    } else {
      tfp <<- import("tensorflow_probability")
      tfp
    }
  })
}

load_tf <- function() {
  tf <- NULL
  makeActiveBinding("tf", env = .GlobalEnv, function() {
    if (!is.null(tf)) {
      return(tf)
    } else {
      tf <<- import("tensorflow")
      tf
    }
  })
}

test_succeeds("can use layer_multivariate_normal_tril in a keras model", {
  
  skip_if_no_tfp(required_version = "0.7")
  load_tf()
  load_tfp()
  
  n <- as.integer(1e3)
  scale_tril <-
    matrix(c(1.6180, 0., -2.7183, 3.1416),
           ncol = 2,
           byrow = TRUE) %>% k_cast_to_floatx()
  scale_noise <- 0.01
  x <- tfp$distributions$Normal(loc = 0, scale = 1)$sample(c(n, 2L))
  eps <-
    tfp$distributions$Normal(loc = 0, scale = scale_noise)$sample(c(1000L, 2L))
  y = tf$matmul(x, scale_tril) + eps
  d <- tf$compat$dimension_value(y$shape[-1])
  
  model <- keras_model_sequential() %>%
    layer_dense(
      units = tfp$layers$MultivariateNormalTriL$params_size(d),
      input_shape = x$shape[-1]
    ) %>%
    layer_multivariate_normal_tril(event_size = d)
  
  log_loss <- function (y, model)
    - model$log_prob(x)
  
  model %>% compile(optimizer = "adam",
                    loss = log_loss)
  
  model %>% fit(
    x,
    y,
    batch_size = 100,
    epochs = 1,
    steps_per_epoch = n / 100
  )
})

test_succeeds("can use layer_kl_divergence_add_loss in a keras model", {
  
  skip_if_no_tfp(required_version = "0.7")
  load_tf()
  load_tfp()
  
  encoded_size <- 2
  input_shape <- c(2L, 2L, 1L)
  train_size <- 100
  x_train <-
    array(runif(train_size * Reduce(`*`, input_shape)), dim = c(train_size, input_shape))
  
  encoder_model <- keras_model_sequential() %>%
    layer_flatten(input_shape = input_shape) %>%
    layer_dense(units = 10, activation = "relu") %>%
    layer_dense(units = tfp$layers$MultivariateNormalTriL$params_size(encoded_size)) %>%
    layer_multivariate_normal_tril(event_size = encoded_size) %>%
    layer_kl_divergence_add_loss(
      distribution = tfp$distributions$Independent(tfp$distributions$Normal(loc = c(0, 0), scale = 1),
                                     reinterpreted_batch_ndims = 1L),
      weight = train_size
    )
  
  decoder_model <- keras_model_sequential() %>%
    layer_dense(units = 10,
                activation = 'relu',
                input_shape = encoded_size) %>%
    layer_dense(tfp$layers$IndependentBernoulli$params_size(input_shape)) %>%
    layer_independent_bernoulli(event_shape = input_shape,
                                convert_to_tensor_fn = tfp$distributions$Bernoulli$logits)
  
  vae_model <- keras_model(inputs = encoder_model$inputs,
                           outputs = decoder_model(encoder_model$outputs[1]))
  
  vae_loss <- function (x, rv_x)
    - rv_x$log_prob(x)
  
  vae_model %>% compile(optimizer = tf$train$AdamOptimizer(),
                        loss = vae_loss)
  
  vae_model %>% fit(x_train,
                    x_train,
                    batch_size = 25,
                    epochs = 1)
  
})

test_succeeds("can use layer_independent_bernoulli in a keras model", {
  
  skip_if_no_tfp(required_version = "0.7")
  load_tf()
  load_tfp()
  
  n <- as.integer(1e3)
  scale_tril <-
    matrix(c(1.6180, 0., -2.7183, 3.1416),
           ncol = 2,
           byrow = TRUE) %>% k_cast_to_floatx()
  scale_noise <- 0.01
  x <- tfp$distributions$Normal(loc = 0, scale = 1)$sample(c(n, 2L))
  eps <-
    tfp$distributions$Normal(loc = 0, scale = scale_noise)$sample(c(1000L, 2L))
  y <-
    tfp$distributions$Bernoulli(logits = tf$reshape(tf$matmul(x, scale_tril) + eps,
                                      shape = shape(n, 1L, 2L, 1L)))$sample()
  
  event_shape <- dim(y)[2:4]
  
  model <- keras_model_sequential() %>%
    layer_dense(
      units = tfp$layers$IndependentBernoulli$params_size(event_shape),
      input_shape = dim(x)[2]
    ) %>%
    layer_independent_bernoulli(event_shape = event_shape)
  
  log_loss <- function (y, model)
    - model$log_prob(y)
  
  model %>% compile(optimizer = "adam",
                    loss = log_loss)
  
  model %>% fit(
    x,
    y,
    batch_size = 100,
    epochs = 1,
    steps_per_epoch = n / 100
  )
  
})
