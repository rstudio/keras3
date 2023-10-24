Applies dropout to the input.

The `Dropout` layer randomly sets input units to 0 with a frequency of
`rate` at each step during training time, which helps prevent overfitting.
Inputs not set to 0 are scaled up by `1 / (1 - rate)` such that the sum over
all inputs is unchanged.

Note that the `Dropout` layer only applies when `training` is set to `True`
in `call()`, such that no values are dropped during inference.
When using `model.fit`, `training` will be appropriately set to `True`
automatically. In other contexts, you can set the argument explicitly
to `True` when calling the layer.

(This is in contrast to setting `trainable=False` for a `Dropout` layer.
`trainable` does not affect the layer's behavior, as `Dropout` does
not have any variables/weights that can be frozen during training.)

Args:
    rate: Float between 0 and 1. Fraction of the input units to drop.
    noise_shape: 1D integer tensor representing the shape of the
        binary dropout mask that will be multiplied with the input.
        For instance, if your inputs have shape
        `(batch_size, timesteps, features)` and
        you want the dropout mask to be the same for all timesteps,
        you can use `noise_shape=(batch_size, 1, features)`.
    seed: A Python integer to use as random seed.

Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
