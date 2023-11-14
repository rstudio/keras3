This wrapper allows to apply a layer to every temporal slice of an input.

@description
Every input should be at least 3D, and the dimension of index one of the
first input will be considered to be the temporal dimension.

Consider a batch of 32 video samples, where each sample is a 128x128 RGB
image with `channels_last` data format, across 10 timesteps.
The batch input shape is `(32, 10, 128, 128, 3)`.

You can then use `TimeDistributed` to apply the same `Conv2D` layer to each
of the 10 timesteps, independently:


```r
inputs <- layer_input(shape = c(10, 128, 128, 3), batch_size = 32)
conv_2d_layer <- layer_conv_2d(filters = 64, kernel_size = c(3, 3))
outputs <- layer_time_distributed(inputs, layer = conv_2d_layer)
outputs$shape
```

```
## [[1]]
## [1] 32
##
## [[2]]
## [1] 10
##
## [[3]]
## [1] 126
##
## [[4]]
## [1] 126
##
## [[5]]
## [1] 64
```

Because `layer_time_distributed` applies the same instance of `layer_conv2d` to each of
the timestamps, the same set of weights are used at each timestamp.

# Call Arguments
- `inputs`: Input tensor of shape (batch, time, ...) or nested tensors,
    and each of which has shape (batch, time, ...).
- `training`: Boolean indicating whether the layer should behave in
    training mode or in inference mode. This argument is passed to the
    wrapped layer (only if the layer supports this argument).
- `mask`: Binary tensor of shape `(samples, timesteps)` indicating whether
    a given timestep should be masked. This argument is passed to the
    wrapped layer (only if the layer supports this argument).

@param layer
a `layer_Layer` instance.

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
Passed on to the Python callable

@export
@family distributed time rnn layers
@family time rnn layers
@family rnn layers
@family layers
@seealso
+ <https:/keras.io/api/layers/recurrent_layers/time_distributed#timedistributed-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed>

