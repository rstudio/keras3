keras.ops.logspace
__signature__
(
  start,
  stop,
  num=50,
  endpoint=True,
  base=10,
  dtype=None,
  axis=0
)
__doc__
Returns numbers spaced evenly on a log scale.

In linear space, the sequence starts at `base ** start` and ends with
`base ** stop` (see `endpoint` below).

Args:
    start: The starting value of the sequence.
    stop: The final value of the sequence, unless `endpoint` is `False`.
        In that case, `num + 1` values are spaced over the interval in
        log-space, of which all but the last (a sequence of length `num`)
        are returned.
    num: Number of samples to generate. Defaults to `50`.
    endpoint: If `True`, `stop` is the last sample. Otherwise, it is not
        included. Defaults to`True`.
    base: The base of the log space. Defaults to `10`.
    dtype: The type of the output tensor.
    axis: The axis in the result to store the samples. Relevant only
        if start or stop are array-like.

Note:
    Torch backend does not support `axis` argument.

Returns:
    A tensor of evenly spaced samples on a log scale.
