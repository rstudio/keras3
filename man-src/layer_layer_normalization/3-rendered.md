Layer normalization layer (Ba et al., 2016).

@description
Normalize the activations of the previous layer for each given example in a
batch independently, rather than across a batch like Batch Normalization.
i.e. applies a transformation that maintains the mean activation within each
example close to 0 and the activation standard deviation close to 1.

If `scale` or `center` are enabled, the layer will scale the normalized
outputs by broadcasting them with a trainable variable `gamma`, and center
the outputs by broadcasting with a trainable variable `beta`. `gamma` will
default to a ones tensor and `beta` will default to a zeros tensor, so that
centering and scaling are no-ops before training has begun.

So, with scaling and centering enabled the normalization equations
are as follows:

Let the intermediate activations for a mini-batch to be the `inputs`.

For each sample `x` in a batch of `inputs`, we compute the mean and
variance of the sample, normalize each value in the sample
(including a small factor `epsilon` for numerical stability),
and finally,
transform the normalized output by `gamma` and `beta`,
which are learned parameters:


```r
outputs <- inputs |> apply(1, function(x) {
  x_normalized <- (x - mean(x)) /
                  sqrt(var(x) + epsilon)
  x_normalized * gamma + beta
})
```

`gamma` and `beta` will span the axes of `inputs` specified in `axis`, and
this part of the inputs' shape must be fully defined.

For example:


```r
layer <- layer_layer_normalization(axis = c(2, 3, 4))

layer(k_ones(c(5, 20, 30, 40))) |> invisible() # build()
shape(layer$beta)
```

```
## shape(20, 30, 40)
```

```r
shape(layer$gamma)
```

```
## shape(20, 30, 40)
```

Note that other implementations of layer normalization may choose to define
`gamma` and `beta` over a separate set of axes from the axes being
normalized across. For example, Group Normalization
([Wu et al. 2018](https://arxiv.org/abs/1803.08494)) with group size of 1
corresponds to a `layer_layer_normalization()` that normalizes across height, width,
and channel and has `gamma` and `beta` span only the channel dimension.
So, this `layer_layer_normalization()` implementation will not match a
`layer_group_normalization()` layer with group size set to 1.

# Reference
- [Lei Ba et al., 2016](https://arxiv.org/abs/1607.06450).

@param axis
Integer or list. The axis or axes to normalize across.
Typically, this is the features axis/axes. The left-out axes are
typically the batch axis/axes. `-1` is the last dimension in the
input. Defaults to `-1`.

@param epsilon
Small float added to variance to avoid dividing by zero.
Defaults to 1e-3.

@param center
If `TRUE`, add offset of `beta` to normalized tensor. If `FALSE`,
`beta` is ignored. Defaults to `TRUE`.

@param scale
If `TRUE`, multiply by `gamma`. If `FALSE`, `gamma` is not used.
When the next layer is linear (also e.g. `layer_activation_relu()`), this can be
disabled since the scaling will be done by the next layer.
Defaults to `TRUE`.

@param rms_scaling
If `TRUE`, `center` and `scale` are ignored, and the
inputs are scaled by `gamma` and the inverse square root
of the square of all inputs. This is an approximate and faster
approach that avoids ever computing the mean of the input.

@param beta_initializer
Initializer for the beta weight. Defaults to zeros.

@param gamma_initializer
Initializer for the gamma weight. Defaults to ones.

@param beta_regularizer
Optional regularizer for the beta weight.
`NULL` by default.

@param gamma_regularizer
Optional regularizer for the gamma weight.
`NULL` by default.

@param beta_constraint
Optional constraint for the beta weight.
`NULL` by default.

@param gamma_constraint
Optional constraint for the gamma weight.
`NULL` by default.

@param ...
Base layer keyword arguments (e.g. `name` and `dtype`).

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@export
@family normalization layers
@family layers
@seealso
+ <https:/keras.io/api/layers/normalization_layers/layer_normalization#layernormalization-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization>
