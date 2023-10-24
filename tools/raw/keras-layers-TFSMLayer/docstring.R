Reload a Keras model/layer that was saved via SavedModel / ExportArchive.

Arguments:
    filepath: `str` or `pathlib.Path` object. The path to the SavedModel.
    call_endpoint: Name of the endpoint to use as the `call()` method
        of the reloaded layer. If the SavedModel was created
        via `model.export()`,
        then the default endpoint name is `'serve'`. In other cases
        it may be named `'serving_default'`.

Example:

```python
model.export("path/to/artifact")
reloaded_layer = TFSMLayer("path/to/artifact")
outputs = reloaded_layer(inputs)
```

The reloaded object can be used like a regular Keras layer, and supports
training/fine-tuning of its trainable weights. Note that the reloaded
object retains none of the internal structure or custom methods of the
original object -- it's a brand new layer created around the saved
function.

**Limitations:**

* Only call endpoints with a single `inputs` tensor argument
(which may optionally be a dict/tuple/list of tensors) are supported.
For endpoints with multiple separate input tensor arguments, consider
subclassing `TFSMLayer` and implementing a `call()` method with a
custom signature.
* If you need training-time behavior to differ from inference-time behavior
(i.e. if you need the reloaded object to support a `training=True` argument
in `__call__()`), make sure that the training-time call function is
saved as a standalone endpoint in the artifact, and provide its name
to the `TFSMLayer` via the `call_training_endpoint` argument.
