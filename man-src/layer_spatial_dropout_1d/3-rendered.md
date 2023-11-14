Spatial 1D version of Dropout.

@description
This layer performs the same function as Dropout, however, it drops
entire 1D feature maps instead of individual elements. If adjacent frames
within feature maps are strongly correlated (as is normally the case in
early convolution layers) then regular dropout will not regularize the
activations and will otherwise just result in an effective learning rate
decrease. In this case, `SpatialDropout1D` will help promote independence
between feature maps and should be used instead.

# Call Arguments
- `inputs`: A 3D tensor.
- `training`: Python boolean indicating whether the layer
    should behave in training mode (applying dropout)
    or in inference mode (pass-through).

# Input Shape
3D tensor with shape: `(samples, timesteps, channels)`

# Output Shape
Same as input.

# Reference
- [Tompson et al., 2014](https://arxiv.org/abs/1411.4280)

@param rate
Float between 0 and 1. Fraction of the input units to drop.

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param name
String, name for the object

@param seed
Initial seed for the random number generator

@param dtype
datatype (e.g., `"float32"`).

@export
@family dropout spatial regularization layers
@family spatial regularization layers
@family regularization layers
@family layers
@seealso
+ <https:/keras.io/api/layers/regularization_layers/spatial_dropout1d#spatialdropout1d-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/SpatialDropout1D>
