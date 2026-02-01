# Keras Layer that wraps a JAX model.

This layer enables the use of JAX components within Keras when using JAX
as the backend for Keras.

## Usage

``` r
layer_jax_model_wrapper(
  object,
  call_fn,
  init_fn = NULL,
  params = NULL,
  state = NULL,
  seed = NULL,
  ...,
  dtype = NULL
)
```

## Arguments

- object:

  Object to compose the layer with. A tensor, array, or sequential
  model.

- call_fn:

  The function to call the model. See description above for the list of
  arguments it takes and the outputs it returns.

- init_fn:

  the function to call to initialize the model. See description above
  for the list of arguments it takes and the outputs it returns. If
  `NULL`, then `params` and/or `state` must be provided.

- params:

  A `PyTree` containing all the model trainable parameters. This allows
  passing trained parameters or controlling the initialization. If both
  `params` and `state` are `NULL`, `init_fn()` is called at build time
  to initialize the trainable parameters of the model.

- state:

  A `PyTree` containing all the model non-trainable state. This allows
  passing learned state or controlling the initialization. If both
  `params` and `state` are `NULL`, and `call_fn()` takes a `state`
  argument, then `init_fn()` is called at build time to initialize the
  non-trainable state of the model.

- seed:

  Seed for random number generator. Optional.

- ...:

  For forward/backward compatability.

- dtype:

  The dtype of the layer's computations and weights. Can also be a
  `keras.DTypePolicy`. Optional. Defaults to the default policy.

## Value

The return value depends on the value provided for the first argument.
If `object` is:

- a
  [`keras_model_sequential()`](https://keras3.posit.co/reference/keras_model_sequential.md),
  then the layer is added to the sequential model (which is modified in
  place). To enable piping, the sequential model is also returned,
  invisibly.

- a [`keras_input()`](https://keras3.posit.co/reference/keras_input.md),
  then the output tensor from calling `layer(input)` is returned.

- `NULL` or missing, then a `Layer` instance is returned.

## Model function

This layer accepts JAX models in the form of a function, `call_fn()`,
which must take the following arguments with these exact names:

- `params`: trainable parameters of the model.

- `state` (*optional*): non-trainable state of the model. Can be omitted
  if the model has no non-trainable state.

- `rng` (*optional*): a `jax.random.PRNGKey` instance. Can be omitted if
  the model does not need RNGs, neither during training nor during
  inference.

- `inputs`: inputs to the model, a JAX array or a `PyTree` of arrays.

- `training` (*optional*): an argument specifying if we're in training
  mode or inference mode, `TRUE` is passed in training mode. Can be
  omitted if the model behaves the same in training mode and inference
  mode.

The `inputs` argument is mandatory. Inputs to the model must be provided
via a single argument. If the JAX model takes multiple inputs as
separate arguments, they must be combined into a single structure, for
instance in a
[`tuple()`](https://rstudio.github.io/reticulate/reference/tuple.html)
or a
[`dict()`](https://rstudio.github.io/reticulate/reference/dict.html).

### Model weights initialization

The initialization of the `params` and `state` of the model can be
handled by this layer, in which case the `init_fn()` argument must be
provided. This allows the model to be initialized dynamically with the
right shape. Alternatively, and if the shape is known, the `params`
argument and optionally the `state` argument can be used to create an
already initialized model.

The `init_fn()` function, if provided, must take the following arguments
with these exact names:

- `rng`: a `jax.random.PRNGKey` instance.

- `inputs`: a JAX array or a `PyTree` of arrays with placeholder values
  to provide the shape of the inputs.

- `training` (*optional*): an argument specifying if we're in training
  mode or inference mode. `True` is always passed to `init_fn`. Can be
  omitted regardless of whether `call_fn` has a `training` argument.

### Models with non-trainable state

For JAX models that have non-trainable state:

- `call_fn()` must have a `state` argument

- `call_fn()` must return a
  [`tuple()`](https://rstudio.github.io/reticulate/reference/tuple.html)
  containing the outputs of the model and the new non-trainable state of
  the model

- `init_fn()` must return a
  [`tuple()`](https://rstudio.github.io/reticulate/reference/tuple.html)
  containing the initial trainable params of the model and the initial
  non-trainable state of the model.

This code shows a possible combination of `call_fn()` and `init_fn()`
signatures for a model with non-trainable state. In this example, the
model has a `training` argument and an `rng` argument in `call_fn()`.

    stateful_call <- function(params, state, rng, inputs, training) {
      outputs <- ....
      new_state <- ....
      tuple(outputs, new_state)
    }

    stateful_init <- function(rng, inputs) {
      initial_params <- ....
      initial_state <- ....
      tuple(initial_params, initial_state)
    }

### Models without non-trainable state

For JAX models with no non-trainable state:

- `call_fn()` must not have a `state` argument

- `call_fn()` must return only the outputs of the model

- `init_fn()` must return only the initial trainable params of the
  model.

This code shows a possible combination of `call_fn()` and `init_fn()`
signatures for a model without non-trainable state. In this example, the
model does not have a `training` argument and does not have an `rng`
argument in `call_fn()`.

    stateful_call <- function(pparams, inputs) {
      outputs <- ....
      outputs
    }

    stateful_init <- function(rng, inputs) {
      initial_params <- ....
      initial_params
    }

### Conforming to the required signature

If a model has a different signature than the one required by
`JaxLayer`, one can easily write a wrapper method to adapt the
arguments. This example shows a model that has multiple inputs as
separate arguments, expects multiple RNGs in a `dict`, and has a
`deterministic` argument with the opposite meaning of `training`. To
conform, the inputs are combined in a single structure using a `tuple`,
the RNG is split and used the populate the expected `dict`, and the
Boolean flag is negated:

    jax <- import("jax")
    my_model_fn <- function(params, rngs, input1, input2, deterministic) {
      ....
      if (!deterministic) {
        dropout_rng <- rngs$dropout
        keep <- jax$random$bernoulli(dropout_rng, dropout_rate, x$shape)
        x <- jax$numpy$where(keep, x / dropout_rate, 0)
        ....
      }
      ....
      return(outputs)
    }

    my_model_wrapper_fn <- function(params, rng, inputs, training) {
      c(input1, input2) %<-% inputs
      c(rng1, rng2) %<-% jax$random$split(rng)
      rngs <-  list(dropout = rng1, preprocessing = rng2)
      deterministic <-  !training
      my_model_fn(params, rngs, input1, input2, deterministic)
    }

    keras_layer <- layer_jax_model_wrapper(call_fn = my_model_wrapper_fn,
                                           params = initial_params)

### Usage with Haiku modules

`JaxLayer` enables the use of [Haiku](https://dm-haiku.readthedocs.io)
components in the form of
[`haiku.Module`](https://dm-haiku.readthedocs.io/en/latest/api.html#module).
This is achieved by transforming the module per the Haiku pattern and
then passing `module.apply` in the `call_fn` parameter and `module.init`
in the `init_fn` parameter if needed.

If the model has non-trainable state, it should be transformed with
[`haiku.transform_with_state`](https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.transform_with_state).
If the model has no non-trainable state, it should be transformed with
[`haiku.transform`](https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.transform).
Additionally, and optionally, if the module does not use RNGs in
"apply", it can be transformed with
[`haiku.without_apply_rng`](https://dm-haiku.readthedocs.io/en/latest/api.html#without-apply-rng).

The following example shows how to create a `JaxLayer` from a Haiku
module that uses random number generators via `hk.next_rng_key()` and
takes a training positional argument:

    # reticulate::py_install("haiku", "r-keras")
    hk <- import("haiku")
    MyHaikuModule(hk$Module) \%py_class\% {

      `__call__` <- \(self, x, training) {
        x <- hk$Conv2D(32L, tuple(3L, 3L))(x)
        x <- jax$nn$relu(x)
        x <- hk$AvgPool(tuple(1L, 2L, 2L, 1L),
                        tuple(1L, 2L, 2L, 1L), "VALID")(x)
        x <- hk$Flatten()(x)
        x <- hk$Linear(200L)(x)
        if (training)
          x <- hk$dropout(rng = hk$next_rng_key(), rate = 0.3, x = x)
        x <- jax$nn$relu(x)
        x <- hk$Linear(10L)(x)
        x <- jax$nn$softmax(x)
        x
      }

    }

    my_haiku_module_fn <- function(inputs, training) {
      module <- MyHaikuModule()
      module(inputs, training)
    }

    transformed_module <- hk$transform(my_haiku_module_fn)

    keras_layer <-
      layer_jax_model_wrapper(call_fn = transformed_module$apply,
                              init_fn = transformed_module$init)

## See also

Other wrapping layers:  
[`layer_flax_module_wrapper()`](https://keras3.posit.co/reference/layer_flax_module_wrapper.md)  
[`layer_torch_module_wrapper()`](https://keras3.posit.co/reference/layer_torch_module_wrapper.md)  

Other layers:  
[`Layer()`](https://keras3.posit.co/reference/Layer.md)  
[`layer_activation()`](https://keras3.posit.co/reference/layer_activation.md)  
[`layer_activation_elu()`](https://keras3.posit.co/reference/layer_activation_elu.md)  
[`layer_activation_leaky_relu()`](https://keras3.posit.co/reference/layer_activation_leaky_relu.md)  
[`layer_activation_parametric_relu()`](https://keras3.posit.co/reference/layer_activation_parametric_relu.md)  
[`layer_activation_relu()`](https://keras3.posit.co/reference/layer_activation_relu.md)  
[`layer_activation_softmax()`](https://keras3.posit.co/reference/layer_activation_softmax.md)  
[`layer_activity_regularization()`](https://keras3.posit.co/reference/layer_activity_regularization.md)  
[`layer_add()`](https://keras3.posit.co/reference/layer_add.md)  
[`layer_additive_attention()`](https://keras3.posit.co/reference/layer_additive_attention.md)  
[`layer_alpha_dropout()`](https://keras3.posit.co/reference/layer_alpha_dropout.md)  
[`layer_attention()`](https://keras3.posit.co/reference/layer_attention.md)  
[`layer_aug_mix()`](https://keras3.posit.co/reference/layer_aug_mix.md)  
[`layer_auto_contrast()`](https://keras3.posit.co/reference/layer_auto_contrast.md)  
[`layer_average()`](https://keras3.posit.co/reference/layer_average.md)  
[`layer_average_pooling_1d()`](https://keras3.posit.co/reference/layer_average_pooling_1d.md)  
[`layer_average_pooling_2d()`](https://keras3.posit.co/reference/layer_average_pooling_2d.md)  
[`layer_average_pooling_3d()`](https://keras3.posit.co/reference/layer_average_pooling_3d.md)  
[`layer_batch_normalization()`](https://keras3.posit.co/reference/layer_batch_normalization.md)  
[`layer_bidirectional()`](https://keras3.posit.co/reference/layer_bidirectional.md)  
[`layer_category_encoding()`](https://keras3.posit.co/reference/layer_category_encoding.md)  
[`layer_center_crop()`](https://keras3.posit.co/reference/layer_center_crop.md)  
[`layer_concatenate()`](https://keras3.posit.co/reference/layer_concatenate.md)  
[`layer_conv_1d()`](https://keras3.posit.co/reference/layer_conv_1d.md)  
[`layer_conv_1d_transpose()`](https://keras3.posit.co/reference/layer_conv_1d_transpose.md)  
[`layer_conv_2d()`](https://keras3.posit.co/reference/layer_conv_2d.md)  
[`layer_conv_2d_transpose()`](https://keras3.posit.co/reference/layer_conv_2d_transpose.md)  
[`layer_conv_3d()`](https://keras3.posit.co/reference/layer_conv_3d.md)  
[`layer_conv_3d_transpose()`](https://keras3.posit.co/reference/layer_conv_3d_transpose.md)  
[`layer_conv_lstm_1d()`](https://keras3.posit.co/reference/layer_conv_lstm_1d.md)  
[`layer_conv_lstm_2d()`](https://keras3.posit.co/reference/layer_conv_lstm_2d.md)  
[`layer_conv_lstm_3d()`](https://keras3.posit.co/reference/layer_conv_lstm_3d.md)  
[`layer_cropping_1d()`](https://keras3.posit.co/reference/layer_cropping_1d.md)  
[`layer_cropping_2d()`](https://keras3.posit.co/reference/layer_cropping_2d.md)  
[`layer_cropping_3d()`](https://keras3.posit.co/reference/layer_cropping_3d.md)  
[`layer_cut_mix()`](https://keras3.posit.co/reference/layer_cut_mix.md)  
[`layer_dense()`](https://keras3.posit.co/reference/layer_dense.md)  
[`layer_depthwise_conv_1d()`](https://keras3.posit.co/reference/layer_depthwise_conv_1d.md)  
[`layer_depthwise_conv_2d()`](https://keras3.posit.co/reference/layer_depthwise_conv_2d.md)  
[`layer_discretization()`](https://keras3.posit.co/reference/layer_discretization.md)  
[`layer_dot()`](https://keras3.posit.co/reference/layer_dot.md)  
[`layer_dropout()`](https://keras3.posit.co/reference/layer_dropout.md)  
[`layer_einsum_dense()`](https://keras3.posit.co/reference/layer_einsum_dense.md)  
[`layer_embedding()`](https://keras3.posit.co/reference/layer_embedding.md)  
[`layer_equalization()`](https://keras3.posit.co/reference/layer_equalization.md)  
[`layer_feature_space()`](https://keras3.posit.co/reference/layer_feature_space.md)  
[`layer_flatten()`](https://keras3.posit.co/reference/layer_flatten.md)  
[`layer_flax_module_wrapper()`](https://keras3.posit.co/reference/layer_flax_module_wrapper.md)  
[`layer_gaussian_dropout()`](https://keras3.posit.co/reference/layer_gaussian_dropout.md)  
[`layer_gaussian_noise()`](https://keras3.posit.co/reference/layer_gaussian_noise.md)  
[`layer_global_average_pooling_1d()`](https://keras3.posit.co/reference/layer_global_average_pooling_1d.md)  
[`layer_global_average_pooling_2d()`](https://keras3.posit.co/reference/layer_global_average_pooling_2d.md)  
[`layer_global_average_pooling_3d()`](https://keras3.posit.co/reference/layer_global_average_pooling_3d.md)  
[`layer_global_max_pooling_1d()`](https://keras3.posit.co/reference/layer_global_max_pooling_1d.md)  
[`layer_global_max_pooling_2d()`](https://keras3.posit.co/reference/layer_global_max_pooling_2d.md)  
[`layer_global_max_pooling_3d()`](https://keras3.posit.co/reference/layer_global_max_pooling_3d.md)  
[`layer_group_normalization()`](https://keras3.posit.co/reference/layer_group_normalization.md)  
[`layer_group_query_attention()`](https://keras3.posit.co/reference/layer_group_query_attention.md)  
[`layer_gru()`](https://keras3.posit.co/reference/layer_gru.md)  
[`layer_hashed_crossing()`](https://keras3.posit.co/reference/layer_hashed_crossing.md)  
[`layer_hashing()`](https://keras3.posit.co/reference/layer_hashing.md)  
[`layer_identity()`](https://keras3.posit.co/reference/layer_identity.md)  
[`layer_integer_lookup()`](https://keras3.posit.co/reference/layer_integer_lookup.md)  
[`layer_lambda()`](https://keras3.posit.co/reference/layer_lambda.md)  
[`layer_layer_normalization()`](https://keras3.posit.co/reference/layer_layer_normalization.md)  
[`layer_lstm()`](https://keras3.posit.co/reference/layer_lstm.md)  
[`layer_masking()`](https://keras3.posit.co/reference/layer_masking.md)  
[`layer_max_num_bounding_boxes()`](https://keras3.posit.co/reference/layer_max_num_bounding_boxes.md)  
[`layer_max_pooling_1d()`](https://keras3.posit.co/reference/layer_max_pooling_1d.md)  
[`layer_max_pooling_2d()`](https://keras3.posit.co/reference/layer_max_pooling_2d.md)  
[`layer_max_pooling_3d()`](https://keras3.posit.co/reference/layer_max_pooling_3d.md)  
[`layer_maximum()`](https://keras3.posit.co/reference/layer_maximum.md)  
[`layer_mel_spectrogram()`](https://keras3.posit.co/reference/layer_mel_spectrogram.md)  
[`layer_minimum()`](https://keras3.posit.co/reference/layer_minimum.md)  
[`layer_mix_up()`](https://keras3.posit.co/reference/layer_mix_up.md)  
[`layer_multi_head_attention()`](https://keras3.posit.co/reference/layer_multi_head_attention.md)  
[`layer_multiply()`](https://keras3.posit.co/reference/layer_multiply.md)  
[`layer_normalization()`](https://keras3.posit.co/reference/layer_normalization.md)  
[`layer_permute()`](https://keras3.posit.co/reference/layer_permute.md)  
[`layer_rand_augment()`](https://keras3.posit.co/reference/layer_rand_augment.md)  
[`layer_random_brightness()`](https://keras3.posit.co/reference/layer_random_brightness.md)  
[`layer_random_color_degeneration()`](https://keras3.posit.co/reference/layer_random_color_degeneration.md)  
[`layer_random_color_jitter()`](https://keras3.posit.co/reference/layer_random_color_jitter.md)  
[`layer_random_contrast()`](https://keras3.posit.co/reference/layer_random_contrast.md)  
[`layer_random_crop()`](https://keras3.posit.co/reference/layer_random_crop.md)  
[`layer_random_elastic_transform()`](https://keras3.posit.co/reference/layer_random_elastic_transform.md)  
[`layer_random_erasing()`](https://keras3.posit.co/reference/layer_random_erasing.md)  
[`layer_random_flip()`](https://keras3.posit.co/reference/layer_random_flip.md)  
[`layer_random_gaussian_blur()`](https://keras3.posit.co/reference/layer_random_gaussian_blur.md)  
[`layer_random_grayscale()`](https://keras3.posit.co/reference/layer_random_grayscale.md)  
[`layer_random_hue()`](https://keras3.posit.co/reference/layer_random_hue.md)  
[`layer_random_invert()`](https://keras3.posit.co/reference/layer_random_invert.md)  
[`layer_random_perspective()`](https://keras3.posit.co/reference/layer_random_perspective.md)  
[`layer_random_posterization()`](https://keras3.posit.co/reference/layer_random_posterization.md)  
[`layer_random_rotation()`](https://keras3.posit.co/reference/layer_random_rotation.md)  
[`layer_random_saturation()`](https://keras3.posit.co/reference/layer_random_saturation.md)  
[`layer_random_sharpness()`](https://keras3.posit.co/reference/layer_random_sharpness.md)  
[`layer_random_shear()`](https://keras3.posit.co/reference/layer_random_shear.md)  
[`layer_random_translation()`](https://keras3.posit.co/reference/layer_random_translation.md)  
[`layer_random_zoom()`](https://keras3.posit.co/reference/layer_random_zoom.md)  
[`layer_repeat_vector()`](https://keras3.posit.co/reference/layer_repeat_vector.md)  
[`layer_rescaling()`](https://keras3.posit.co/reference/layer_rescaling.md)  
[`layer_reshape()`](https://keras3.posit.co/reference/layer_reshape.md)  
[`layer_resizing()`](https://keras3.posit.co/reference/layer_resizing.md)  
[`layer_rms_normalization()`](https://keras3.posit.co/reference/layer_rms_normalization.md)  
[`layer_rnn()`](https://keras3.posit.co/reference/layer_rnn.md)  
[`layer_separable_conv_1d()`](https://keras3.posit.co/reference/layer_separable_conv_1d.md)  
[`layer_separable_conv_2d()`](https://keras3.posit.co/reference/layer_separable_conv_2d.md)  
[`layer_simple_rnn()`](https://keras3.posit.co/reference/layer_simple_rnn.md)  
[`layer_solarization()`](https://keras3.posit.co/reference/layer_solarization.md)  
[`layer_spatial_dropout_1d()`](https://keras3.posit.co/reference/layer_spatial_dropout_1d.md)  
[`layer_spatial_dropout_2d()`](https://keras3.posit.co/reference/layer_spatial_dropout_2d.md)  
[`layer_spatial_dropout_3d()`](https://keras3.posit.co/reference/layer_spatial_dropout_3d.md)  
[`layer_spectral_normalization()`](https://keras3.posit.co/reference/layer_spectral_normalization.md)  
[`layer_stft_spectrogram()`](https://keras3.posit.co/reference/layer_stft_spectrogram.md)  
[`layer_string_lookup()`](https://keras3.posit.co/reference/layer_string_lookup.md)  
[`layer_subtract()`](https://keras3.posit.co/reference/layer_subtract.md)  
[`layer_text_vectorization()`](https://keras3.posit.co/reference/layer_text_vectorization.md)  
[`layer_tfsm()`](https://keras3.posit.co/reference/layer_tfsm.md)  
[`layer_time_distributed()`](https://keras3.posit.co/reference/layer_time_distributed.md)  
[`layer_torch_module_wrapper()`](https://keras3.posit.co/reference/layer_torch_module_wrapper.md)  
[`layer_unit_normalization()`](https://keras3.posit.co/reference/layer_unit_normalization.md)  
[`layer_upsampling_1d()`](https://keras3.posit.co/reference/layer_upsampling_1d.md)  
[`layer_upsampling_2d()`](https://keras3.posit.co/reference/layer_upsampling_2d.md)  
[`layer_upsampling_3d()`](https://keras3.posit.co/reference/layer_upsampling_3d.md)  
[`layer_zero_padding_1d()`](https://keras3.posit.co/reference/layer_zero_padding_1d.md)  
[`layer_zero_padding_2d()`](https://keras3.posit.co/reference/layer_zero_padding_2d.md)  
[`layer_zero_padding_3d()`](https://keras3.posit.co/reference/layer_zero_padding_3d.md)  
[`rnn_cell_gru()`](https://keras3.posit.co/reference/rnn_cell_gru.md)  
[`rnn_cell_lstm()`](https://keras3.posit.co/reference/rnn_cell_lstm.md)  
[`rnn_cell_simple()`](https://keras3.posit.co/reference/rnn_cell_simple.md)  
[`rnn_cells_stack()`](https://keras3.posit.co/reference/rnn_cells_stack.md)  
