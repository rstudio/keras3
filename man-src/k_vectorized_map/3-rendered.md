Parallel map of `function` on axis 0 of tensor(s) `elements`.

@description
Schematically, `vectorized_map` implements the following,
in the case of a single tensor input `elements`:

```python
def vectorized_map(function, elements)
    outputs = []
    for e in elements:
        outputs.append(function(e))
    return stack(outputs)
```

In the case of an iterable of tensors `elements`,
it implements the following:

```python
def vectorized_map(function, elements)
    batch_size = elements[0].shape[0]
    outputs = []
    for index in range(batch_size):
        outputs.append(function([e[index] for e in elements]))
    return np.stack(outputs)
```

In this case, `function` is expected to take as input
a single list of tensor arguments.

@param elements see description

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/vectorized_map>
