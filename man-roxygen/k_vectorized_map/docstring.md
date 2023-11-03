Parallel map of `function` on axis 0 of tensor `x`.

Schematically, `vectorized_map` implements the following:

```python
def vectorized_map(function, x)
    outputs = []
    for element in x:
        outputs.append(function(element))
    return stack(outputs)
