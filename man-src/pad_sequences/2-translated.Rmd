Pads sequences to the same length.

@description
This function transforms a list (of length `num_samples`)
of sequences (lists of integers)
into a 2D NumPy array of shape `(num_samples, num_timesteps)`.
`num_timesteps` is either the `maxlen` argument if provided,
or the length of the longest sequence in the list.

Sequences that are shorter than `num_timesteps`
are padded with `value` until they are `num_timesteps` long.

Sequences longer than `num_timesteps` are truncated
so that they fit the desired length.

The position where padding or truncation happens is determined by
the arguments `padding` and `truncating`, respectively.
Pre-padding or removing values from the beginning of the sequence is the
default.

```python
sequence = [[1], [2, 3], [4, 5, 6]]
keras.utils.pad_sequences(sequence)
# array([[0, 0, 1],
#        [0, 2, 3],
#        [4, 5, 6]], dtype=int32)
```

```python
keras.utils.pad_sequences(sequence, value=-1)
# array([[-1, -1,  1],
#        [-1,  2,  3],
#        [ 4,  5,  6]], dtype=int32)
```

```python
keras.utils.pad_sequences(sequence, padding='post')
# array([[1, 0, 0],
#        [2, 3, 0],
#        [4, 5, 6]], dtype=int32)
```

```python
keras.utils.pad_sequences(sequence, maxlen=2)
# array([[0, 1],
#        [2, 3],
#        [5, 6]], dtype=int32)
```

@returns
    NumPy array with shape `(len(sequences), maxlen)`

@param sequences List of sequences (each sequence is a list of integers).
@param maxlen Optional Int, maximum length of all sequences. If not provided,
    sequences will be padded to the length of the longest individual
    sequence.
@param dtype (Optional, defaults to `"int32"`). Type of the output sequences.
    To pad sequences with variable length strings, you can use `object`.
@param padding String, "pre" or "post" (optional, defaults to `"pre"`):
    pad either before or after each sequence.
@param truncating String, "pre" or "post" (optional, defaults to `"pre"`):
    remove values from sequences larger than
    `maxlen`, either at the beginning or at the end of the sequences.
@param value Float or String, padding value. (Optional, defaults to 0.)

@export
@family utils
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences>
