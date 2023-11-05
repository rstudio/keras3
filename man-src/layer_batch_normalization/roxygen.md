Layer that normalizes its inputs.

@description
Batch normalization applies a transformation that maintains the mean output
close to 0 and the output standard deviation close to 1.

Importantly, batch normalization works differently during training and
during inference.

**During training** (i.e. when using `fit()` or when calling the layer/model
with the argument `training=True`), the layer normalizes its output using
the mean and standard deviation of the current batch of inputs. That is to
say, for each channel being normalized, the layer returns
`gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta`, where:

- `epsilon` is small constant (configurable as part of the constructor
arguments)
- `gamma` is a learned scaling factor (initialized as 1), which
can be disabled by passing `scale=False` to the constructor.
- `beta` is a learned offset factor (initialized as 0), which
can be disabled by passing `center=False` to the constructor.

**During inference** (i.e. when using `evaluate()` or `predict()` or when
calling the layer/model with the argument `training=False` (which is the
default), the layer normalizes its output using a moving average of the
mean and standard deviation of the batches it has seen during training. That
is to say, it returns
`gamma * (batch - self.moving_mean) / sqrt(self.moving_var+epsilon) + beta`.

`self.moving_mean` and `self.moving_var` are non-trainable variables that
are updated each time the layer in called in training mode, as such:

- `moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)`
- `moving_var = moving_var * momentum + var(batch) * (1 - momentum)`

As such, the layer will only normalize its inputs during inference
*after having been trained on data that has similar statistics as the
inference data*.

**About setting `layer.trainable = False` on a `BatchNormalization` layer:**

The meaning of setting `layer.trainable = False` is to freeze the layer,
i.e. its internal state will not change during training:
its trainable weights will not be updated
during `fit()` or `train_on_batch()`, and its state updates will not be run.

Usually, this does not necessarily mean that the layer is run in inference
mode (which is normally controlled by the `training` argument that can
be passed when calling a layer). "Frozen state" and "inference mode"
are two separate concepts.

However, in the case of the `BatchNormalization` layer, **setting
`trainable = False` on the layer means that the layer will be
subsequently run in inference mode** (meaning that it will use
the moving mean and the moving variance to normalize the current batch,
rather than using the mean and variance of the current batch).

Note that:

- Setting `trainable` on an model containing other layers will recursively
    set the `trainable` value of all inner layers.
- If the value of the `trainable` attribute is changed after calling
    `compile()` on a model, the new value doesn't take effect for this model
    until `compile()` is called again.

# Call Arguments
- `inputs`: Input tensor (of any rank).
- `training`: Python boolean indicating whether the layer should behave in
    training mode or in inference mode.
    - `training=True`: The layer will normalize its inputs using
    the mean and variance of the current batch of inputs.
    - `training=False`: The layer will normalize its inputs using
    the mean and variance of its moving statistics, learned during
    training.

# Reference
- [Ioffe and Szegedy, 2015](https://arxiv.org/abs/1502.03167).

@param axis Integer, the axis that should be normalized
    (typically the features axis). For instance, after a `Conv2D` layer
    with `data_format="channels_first"`, use `axis=1`.
@param momentum Momentum for the moving average.
@param epsilon Small float added to variance to avoid dividing by zero.
@param center If `True`, add offset of `beta` to normalized tensor.
    If `False`, `beta` is ignored.
@param scale If `True`, multiply by `gamma`. If `False`, `gamma` is not used.
    When the next layer is linear this can be disabled
    since the scaling will be done by the next layer.
@param beta_initializer Initializer for the beta weight.
@param gamma_initializer Initializer for the gamma weight.
@param moving_mean_initializer Initializer for the moving mean.
@param moving_variance_initializer Initializer for the moving variance.
@param beta_regularizer Optional regularizer for the beta weight.
@param gamma_regularizer Optional regularizer for the gamma weight.
@param beta_constraint Optional constraint for the beta weight.
@param gamma_constraint Optional constraint for the gamma weight.
@param synchronized Only applicable with the TensorFlow backend.
    If `True`, synchronizes the global batch statistics (mean and
    variance) for the layer across all devices at each training step
    in a distributed training strategy.
    If `False`, each replica uses its own local batch statistics.
@param ... Base layer keyword arguments (e.g. `name` and `dtype`).
@param object Object to compose the layer with. A tensor, array, or sequential model.

@export
@family normalization layers
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization>
