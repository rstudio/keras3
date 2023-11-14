Initializer that generates tensors with constant values.

@description
Only scalar values are allowed.
The constant value provided must be convertible to the dtype requested
when calling the initializer.

# Examples
```python
# Standalone usage:
initializer = Constant(10.)
values = initializer(shape=(2, 2))
```

```python
# Usage in a Keras layer:
initializer = Constant(10.)
layer = Dense(3, kernel_initializer=initializer)
```

@param value
A Python scalar.

@export
@family initializer
@seealso
+ <https:/keras.io/api/layers/initializers#constant-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Constant>
