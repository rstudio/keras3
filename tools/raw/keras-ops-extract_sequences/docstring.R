Expands the dimension of last axis into sequences of `sequence_length`.

Slides a window of size `sequence_length` over the last axis of the input
with a stride of `sequence_stride`, replacing the last axis with
`[num_sequences, sequence_length]` sequences.

If the dimension along the last axis is N, the number of sequences can be
computed by:

`num_sequences = 1 + (N - sequence_length) // sequence_stride`

Args:
    x: Input tensor.
    sequence_length: An integer representing the sequences length.
    sequence_stride: An integer representing the sequences hop size.

Returns:
    A tensor of sequences with shape [..., num_sequences, sequence_length].

Example:

>>> x = keras.ops.convert_to_tensor([1, 2, 3, 4, 5, 6])
>>> extract_sequences(x, 3, 2)
array([[1, 2, 3],
   [3, 4, 5]])
