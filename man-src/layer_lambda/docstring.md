Wraps arbitrary expressions as a `Layer` object.

The `Lambda` layer exists so that arbitrary expressions can be used
as a `Layer` when constructing Sequential
and Functional API models. `Lambda` layers are best suited for simple
operations or quick experimentation. For more advanced use cases,
prefer writing new subclasses of `Layer`.

WARNING: `Lambda` layers have (de)serialization limitations!

The main reason to subclass `Layer` instead of using a
`Lambda` layer is saving and inspecting a model. `Lambda` layers
are saved by serializing the Python bytecode, which is fundamentally
non-portable and potentially unsafe.
They should only be loaded in the same environment where
they were saved. Subclassed layers can be saved in a more portable way
by overriding their `get_config()` method. Models that rely on
subclassed Layers are also often easier to visualize and reason about.

Example:

```python
# add a x -> x^2 layer
model.add(Lambda(lambda x: x ** 2))
```

Args:
    function: The function to be evaluated. Takes input tensor as first
        argument.
    output_shape: Expected output shape from function. This argument
        can usually be inferred if not explicitly provided.
        Can be a tuple or function. If a tuple, it only specifies
        the first dimension onward; sample dimension is assumed
        either the same as the input:
        `output_shape = (input_shape[0], ) + output_shape` or,
        the input is `None` and the sample dimension is also `None`:
        `output_shape = (None, ) + output_shape`.
        If a function, it specifies the
        entire shape as a function of the input shape:
        `output_shape = f(input_shape)`.
    mask: Either None (indicating no masking) or a callable with the same
        signature as the `compute_mask` layer method, or a tensor
        that will be returned as output mask regardless
        of what the input is.
    arguments: Optional dictionary of keyword arguments to be passed to the
        function.
