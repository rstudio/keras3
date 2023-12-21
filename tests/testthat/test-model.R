
context("model")



test_succeeds("sequential models can be defined", {
  define_model()
})


test_succeeds("sequential models can be compiled", {
  define_and_compile_model()
})

test_succeeds(required_version = "2.0.7", "models can be cloned", {
  model <- define_model()
  model2 <- clone_model(model)
})


# generate dummy training data
data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)
# storage.mode(labels) <- "integer"

# genereate dummy input data
input <- matrix(rexp(10*784), nrow = 10, ncol = 784)


test_succeeds("models can be fit, evaluated, and used for predictions", {
  model <- define_and_compile_model()
  fit(model, data, labels, verbose = 0)
  evaluate(model, data, labels)
  predict(model, input)
  predict_on_batch(model, input)
  if(tf_version() < "2.6") {
    # model.predict_proba and model.predict_classes removed in 2.6
    expect_warning(predict_proba(model, input), "deprecated")
    expect_warning(predict_classes(model, input), "deprecated")
  }
})


test_succeeds("keras_model_sequential() can accept input_layer args", {
  model <-  keras_model_sequential(input_shape = 784) %>%
    layer_dense(32, kernel_initializer = initializer_ones()) %>%
    layer_activation('relu') %>%
    layer_dense(10) %>%
    layer_activation('softmax') %>%
    compile(loss = loss_binary_crossentropy(),
            optimizer = optimizer_sgd(),
            metrics = 'accuracy')

  fit(model, data, labels, verbose = 0)
  evaluate(model, data, labels)
  predict(model, input)
})


test_succeeds("evaluate function returns a named list", {
  model <- define_and_compile_model()
  fit(model, data, labels)
  result <- evaluate(model, data, labels)
  expect_true(!is.null(names(result)))
})

test_succeeds("models can be tested and trained on batches", {
  model <- define_and_compile_model()
  train_on_batch(model, data, labels)
  test_on_batch(model, data, labels)
})


test_succeeds("models layers can be retrieved by name and index", {
  model <- keras_model_sequential()
  model %>%
    layer_dense(32, input_shape = 784, kernel_initializer = initializer_ones()) %>%
    layer_activation('relu', name = 'first_activation') %>%
    layer_dense(10) %>%
    layer_activation('softmax')

  get_layer(model, name = 'first_activation')
  get_layer(model, index = 1)
})


test_succeeds("models layers can be popped", {
  model <- keras_model_sequential()
  model %>%
    layer_dense(32, input_shape = 784, kernel_initializer = initializer_ones()) %>%
    layer_activation('relu', name = 'first_activation') %>%
    layer_dense(10) %>%
    layer_activation('softmax')

  expect_equal(length(model$layers), 4)
  pop_layer(model)
  expect_equal(length(model$layers), 3)

})

test_succeeds("can call model with R objects", {

  if (!tensorflow::tf_version() >= "1.14") skip("Needs TF >= 1.14")

  model <- keras_model_sequential() %>%
    layer_dense(units = 1, input_shape = 1)

  model(
    tensorflow::tf$convert_to_tensor(
      matrix(runif(10), ncol = 1),
      dtype = tensorflow::tf$float32
    )
  )

  input1 <- layer_input(shape = 1)
  input2 <- layer_input(shape = 1)

  output <- layer_concatenate(list(input1, input2))

  model <- keras_model(list(input1, input2), output)
  l <- lapply(
    list(matrix(runif(10), ncol = 1), matrix(runif(10), ncol = 1)),
    function(x) tensorflow::tf$convert_to_tensor(x, dtype = tensorflow::tf$float32)
  )
  model(l)
})


test_succeeds("layer_input()", {
  # can take dtype = Dtype
  layer_input(shape = 1, dtype = tf$string)

  # can take shape=NA correctly
  shp <- layer_input(shape = c(NA))$shape
  if(inherits(shp, "tensorflow.python.framework.tensor_shape.TensorShape"))
    shp <- shp$as_list()

  expect_identical(shp, list(NULL, NULL))
})


test_succeeds("can call a model with additional arguments", {

  if (tensorflow::tf_version() < "2.0") skip("needs TF > 2")

  model <- keras_model_sequential() %>%
    layer_dropout(rate = 0.99999999)
  expect_equivalent(as.numeric(model(array(1), training = TRUE)), 0)
  expect_equivalent(as.numeric(model(array(1), training = FALSE)), 1)

})

test_succeeds("pass validation_data to model fit", {

  model <- keras_model_sequential() %>%
    layer_dense(units =1, input_shape = 2)

  model %>% compile(loss = "mse", optimizer = "sgd")

  model %>%
    fit(
      matrix(runif(100), ncol = 2), y = runif(50),
      batch_size = 10,
      validation_data = list(matrix(runif(100), ncol = 2), runif(50))
    )

})


test_succeeds("can pass name argument to 'keras_model'", {

  inputs <- layer_input(shape = c(1))

  predictions <- inputs %>%
    layer_dense(units = 1)

  name = 'My_keras_model'
  model <- keras_model(inputs = inputs, outputs = predictions, name = name)
  expect_identical(model$name,name)
})

test_succeeds("can print a sequential model that is not built", {

  model <- keras_model_sequential() |> layer_dense(10)

  expect_no_error(
    print(model)
  )

  expect_output(
    print(model),
    regexp = "unbuilt"
  )

})

test_succeeds("can use a loss function defined in python", {

  model <- define_model()
  pyfun <- reticulate::py_run_string("
import tensorflow as tf
def loss_fn (y_true, y_pred):
  return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

")

  model %>%
    compile(
      loss = pyfun$loss_fn,
      optimizer = "adam"
    )

  # generate dummy training data
  data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
  labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)


  model %>% fit(x = data, y = labels)

})

test_succeeds("regression test for https://github.com/rstudio/keras/issues/1201", {

  if (tensorflow::tf_version() == "2.1")
    skip("don't work in tf2.1")

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 1, activation = 'relu', input_shape = c(1)) %>%
    compile(
      optimizer = 'sgd',
      loss = 'binary_crossentropy'
    )

  generator <- function() {
    list(1, 2)
  }

  expect_warning_if(tensorflow::tf_version() > "2.1", {
    model %>% fit_generator(generator, steps_per_epoch = 1, epochs = 5,
                            validation_data = generator, validation_steps = 1)
  })

})


if(tf_version() >= "2.4")
test_succeeds("can use functional api with dicts", {

# arr <- function (..., mode = "double", gen = seq_len)
#   array(as.vector(gen(prod(unlist(c(...)))), mode = mode), unlist(c(...)))

  ## TODO: consistency of names between input dict keys and input tensor names
  ## is now enforced. E.g., this raises a warning if `sapply(inputs, \(t) t$name)`
  ## differs from `names(inputs)`
  # inputs <- list(
  #   input_tensor_1 = layer_input(c(1), name = "input_tensor_1_name"),
  #   input_tensor_2 = layer_input(c(2), name = "input_tensor_2_name"),
  #   input_tensor_3 = layer_input(c(3), name = "input_tensor_3_name")
  # )

  clear_session()
  inputs <- list(
    input_tensor_1 = layer_input(c(1), name = "input_tensor_1"),
    input_tensor_2 = layer_input(c(2), name = "input_tensor_2"),
    input_tensor_3 = layer_input(c(3), name = "input_tensor_3")
  )

  # with keras3, the 'name' argument here doesn't change the neame of the output tensor,
  # only the layer name
  outputs <- list(
    output_tensor_1 = inputs$input_tensor_1 %>% layer_dense(4, name = "output_tensor_1"),
    output_tensor_2 = inputs$input_tensor_2 %>% layer_dense(5, name = "output_tensor_2"),
    output_tensor_3 = inputs$input_tensor_3 %>% layer_dense(6, name = "output_tensor_3")
  )

  N <- 10
  new_xy <- function() {
    x <- list(
      input_tensor_1 = random_array(N, 1),
      input_tensor_2 = random_array(N, 2),
      input_tensor_3 = random_array(N, 3)
    )

    y <- list(
      output_tensor_1 = random_array(N, 4),
      output_tensor_2 = random_array(N, 5),
      output_tensor_3 = random_array(N, 6)
    )
    list(x, y)
  }


  chk <- function(inputs, outputs, x, y, error = FALSE) {

    model <- keras_model(inputs, outputs) %>%
      compile(loss = lapply(outputs, \(o) loss_mean_squared_error()),
              optimizer = optimizer_adam())

    .chk <- vector("list", 4L)
    names(.chk) <- c("call", "fit", "evaluate", "predict")
    for (nm in names(.chk))
      .chk[[nm]] <- expect_no_error

    if (isTRUE(error)) {
      for (nm in names(.chk))
        .chk[[nm]] <- expect_error
    } else if (!isFALSE(error)) {
      .chk[error] <- list(expect_error)
    }

    .chk$call({
      res <- model(x)
      expect_identical(names(res), names(outputs))
    })

    .chk$fit({
      model %>% fit(x, y, epochs = 1, verbose = FALSE)
      # model$fit(x, y, epochs = 1L, verbose = 0L)
    })

    .chk$evaluate({
      model %>% evaluate(x, y, verbose = FALSE)
    })

    .chk$predict({
      res <- model %>% predict(x, verbose = FALSE)
      expect_identical(names(res), names(outputs))
    })
  }

  c(x, y) %<-% new_xy()

  ## bug upstream in TF w/ Python < 3.12, types.Protocol instance checking
  ## properly backported to Python < 3.12, and errors on tf internal _DictWrapper()
  ## Keras 3 (via deps tf-nightly / torch) can't install on Python 3.12 yet.

  # everything unnamed
  chk(unname(inputs), unname(outputs), unname(x), unname(y))
  chk(unname(inputs), unname(outputs), unname(x)[c(3,1,2)], unname(y)[c(2, 3, 1)], error = TRUE)
  chk(unname(inputs), unname(outputs), unname(x), unname(y)[c(2, 3, 1)], error = c("fit", "evaluate"))
  chk(unname(inputs), unname(outputs), unname(x)[c(3,1,2)], unname(y), error = TRUE)


  # everything named
  # skip("bug upstream in tf._DictWrapper and trace.SupportsTracingProtocol")
  chk(inputs, outputs, x, y)
  # chk(inputs, outputs, x[c(3,1,2)], y[c(2, 3, 1)])


  # model constructed with named outputs,
  # passed unnamed x, named y
  # chk(inputs, outputs, unname(x), y)
  chk(inputs, outputs, unname(x)[c(2,1,3)], y, error = TRUE)


  # model constructed with unnamed outputs,
  # passed names that don't match to output_tensor.name's

  # skip("names mismatch for fit() enforced elsewhere now, different rules")
  chk(unname(inputs), unname(outputs), x, y) #, error = TRUE)

  # model constructed with unnamed outputs,
  # passed names that do not match to output_tensor.name's
  chk(unname(inputs), unname(outputs),
      x = rlang::set_names(x, ~ paste0(.x, "_name")),
      y = rlang::set_names(y, ~ paste0(.x, "_name")),
      error = TRUE)

  # model constructed with named outputs,
  # passed names that match to output_tensor.name's, not output names
  chk(inputs, outputs,
      x = rlang::set_names(x, ~ paste0(.x, "_name")),
      y = rlang::set_names(y, ~ paste0(.x, "_name")),
      error = TRUE)


  # model constructed with named outputs
  # passed unnamed(y) (x can still match positionally)
  chk(inputs, outputs, unname(x), unname(y)) #, error = c("fit", "evaluate"))
  chk(inputs, outputs,        x , unname(y)) #, error = c("fit", "evaluate"))

  # model constructed with named outputs
  # passed unname x, but in wrong order so positional mathcing wrong
  chk(inputs, outputs, unname(x)[c(3,1,2)], unname(y), error = TRUE)

})



test_succeeds("can pass pandas.Series() to fit()", {
  #https://github.com/rstudio/keras/issues/1341
  skip_if(tf_version() >= "2.13")
  n <- 30
  p <- 10

  w <- runif(n)
  y <- runif(n)
  X <- matrix(runif(n * p), ncol = p)

  make_nn <- function() {
    input <- layer_input(p)
    output <- input %>%
      layer_dense(2 * p, activation = "tanh") %>%
      layer_dense(1)
    keras_model(inputs = input, outputs = output)
  }

  nn <- make_nn()

  pd <- reticulate::import("pandas", convert = FALSE)
  w <- pd$Series(w)

  nn %>%
    compile(optimizer = optimizer_adam(0.02), loss = "mse",
            weighted_metrics = list()) %>% # silence warning
    fit(
      x = X,
      y = y,
      sample_weight = w,
      weighted_metrics = list(),
      epochs = 2,
      validation_split = 0.2,
      verbose = 0
    )
})


r"---(
context=<tensorflow.core.function.trace_type.trace_type_builder.InternalTracingContext object at 0x2f6019210>
value=DictWrapper({'output_tensor_1': <tf.Tensor 'Identity:0' shape=(10, 4) dtype=float32>, 'output_tensor_2': <tf.Tensor 'Identity_1:0' shape=(10, 5) dtype=float32>, 'output_tensor_3': <tf.Tensor 'Identity_2:0' shape=(10, 6) dtype=float32>})
trace.SupportsTracingProtocol=<class 'tensorflow.python.types.trace.SupportsTracingProtocol'>
trace=<module 'tensorflow.python.types.trace' from '/Users/tomasz/.virtualenvs/r-keras/lib/python3.10/site-packages/tensorflow/python/types/trace.py'>
Error in py_call_impl(callable, call_args$unnamed, call_args$named) :
  TypeError: this __dict__ descriptor does not support '_DictWrapper' objects
Run `reticulate::py_last_error()` for details.
▆
Browse[1]> reticulate::py_last_error()

── Python Exception Message ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Traceback (most recent call last):
  File "/Users/tomasz/.virtualenvs/r-keras/lib/python3.10/site-packages/keras/src/utils/traceback_utils.py", line 114, in error_handler
    return fn(*args, **kwargs)
  File "/Users/tomasz/.virtualenvs/r-keras/lib/python3.10/site-packages/keras/src/backend/tensorflow/trainer.py", line 506, in predict
    batch_outputs = self.predict_function(data)
  File "/Users/tomasz/.virtualenvs/r-keras/lib/python3.10/site-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/Users/tomasz/.virtualenvs/r-keras/lib/python3.10/site-packages/keras/src/backend/tensorflow/trainer.py", line 212, in one_step_on_data_distributed
    outputs = self.distribute_strategy.run(
  File "/Users/tomasz/.virtualenvs/r-keras/lib/python3.10/site-packages/tensorflow/core/function/trace_type/trace_type_builder.py", line 148, in from_value
    elif isinstance(value, trace.SupportsTracingProtocol):
  File "/Users/tomasz/.virtualenvs/r-keras/lib/python3.10/site-packages/typing_extensions.py", line 603, in __instancecheck__
    val = inspect.getattr_static(instance, attr)
  File "/Users/tomasz/.pyenv/versions/3.10.13/lib/python3.10/inspect.py", line 1743, in getattr_static
    instance_result = _check_instance(obj, attr)
  File "/Users/tomasz/.pyenv/versions/3.10.13/lib/python3.10/inspect.py", line 1690, in _check_instance
    instance_dict = object.__getattribute__(obj, "__dict__")
TypeError: this __dict__ descriptor does not support '_DictWrapper' objects

── R Traceback ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     ▆
  1. ├─global chk(inputs, outputs, x, y)
  2. │ ├─.chk$predict(...)
  3. │ │ └─testthat::expect_no_error(...)
  4. │ │   └─testthat:::expect_no_(...)
  5. │ │     └─testthat:::quasi_capture(enquo(object), NULL, capture)
  6. │ │       ├─testthat (local) .capture(...)
  7. │ │       │ └─rlang::try_fetch(...)
  8. │ │       │   ├─base::tryCatch(...)
  9. │ │       │   │ └─base (local) tryCatchList(expr, classes, parentenv, handlers)
 10. │ │       │   │   └─base (local) tryCatchOne(expr, names, parentenv, handlers[[1L]])
 11. │ │       │   │     └─base (local) doTryCatch(return(expr), name, parentenv, handler)
 12. │ │       │   └─base::withCallingHandlers(...)
 13. │ │       └─rlang::eval_bare(quo_get_expr(.quo), quo_get_env(.quo))
 14. │ └─model %>% predict(x)
 15. ├─stats::predict(., x)
 16. └─keras3:::predict.keras.models.model.Model(., x)
 17.   ├─base::do.call(object$predict, args) at keras/R/model.R:889:3
 18.   └─reticulate (local) `<python.builtin.method>`(...)
 19.     └─reticulate:::py_call_impl(callable, call_args$unnamed, call_args$named)
  )---"
