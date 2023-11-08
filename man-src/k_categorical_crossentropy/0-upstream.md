keras.ops.categorical_crossentropy
__signature__
(
  target,
  output,
  from_logits=False,
  axis=-1
)
__doc__
Computes categorical cross-entropy loss between target and output tensor.

The categorical cross-entropy loss is commonly used in multi-class
classification tasks where each input sample can belong to one of
multiple classes. It measures the dissimilarity
between the target and output probabilities or logits.

Args:
    target: The target tensor representing the true categorical labels.
        Its shape should match the shape of the `output` tensor
        except for the last dimension.
    output: The output tensor representing the predicted probabilities
        or logits. Its shape should match the shape of the `target`
        tensor except for the last dimension.
    from_logits: (optional) Whether `output` is a tensor of logits or
        probabilities.
        Set it to `True` if `output` represents logits; otherwise,
        set it to `False` if `output` represents probabilities.
        Defaults to`False`.
    axis: (optional) The axis along which the categorical cross-entropy
        is computed.
        Defaults to `-1`, which corresponds to the last dimension of
        the tensors.

Returns:
    Integer tensor: The computed categorical cross-entropy loss between
    `target` and `output`.

Example:

>>> target = keras.ops.convert_to_tensor(
... [[1, 0, 0],
...  [0, 1, 0],
...  [0, 0, 1]])
>>> output = keras.ops.convert_to_tensor(
... [[0.9, 0.05, 0.05],
...  [0.1, 0.8, 0.1],
...  [0.2, 0.3, 0.5]])
>>> categorical_crossentropy(target, output)
array([0.10536054 0.22314355 0.6931472 ], shape=(3,), dtype=float32)
