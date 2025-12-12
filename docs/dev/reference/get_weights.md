# Layer/Model weights as R arrays

Layer/Model weights as R arrays

## Usage

``` r
get_weights(object, trainable = NA)

set_weights(object, weights)
```

## Arguments

- object:

  Layer or model object

- trainable:

  if `NA` (the default), all weights are returned. If `TRUE`, only
  weights of trainable variables are returned. If `FALSE`, only weights
  of non-trainable variables are returned.

- weights:

  Weights as R array

## Value

A list of R arrays.

## Note

You can access the Layer/Model as `KerasVariables` (which are also
backend-native tensors like `tf.Variable`) at `object$weights`,
`object$trainable_weights`, or `object$non_trainable_weights`

## See also

Other layer methods:  
[`count_params()`](https://keras3.posit.co/dev/reference/count_params.md)  
[`get_config()`](https://keras3.posit.co/dev/reference/get_config.md)  
[`quantize_weights()`](https://keras3.posit.co/dev/reference/quantize_weights.md)  
[`reset_state()`](https://keras3.posit.co/dev/reference/reset_state.md)  
