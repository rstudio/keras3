Apply multiplicative 1-centered Gaussian noise.

@description
As it is a regularization layer, it is only active at training time.

# Call Arguments
- `inputs`: Input tensor (of any rank).
- `training`: Python boolean indicating whether the layer should behave in
    training mode (adding dropout) or in inference mode (doing nothing).

@param rate
Float, drop probability (as with `Dropout`).
The multiplicative noise will have
standard deviation `sqrt(rate / (1 - rate))`.

@param seed
Integer, optional random seed to enable deterministic behavior.

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
Passed on to the Python callable

@export
@family regularization layers
@seealso
+ <https:/keras.io/api/layers/regularization_layers/gaussian_dropout#gaussiandropout-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GaussianDropout>
