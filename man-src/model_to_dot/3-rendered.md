Convert a Keras model to dot format.

@returns
A `pydot.Dot` instance representing the Keras model or
a `pydot.Cluster` instance representing nested model if
`subgraph=True`.

@param model A Keras model instance.
@param show_shapes whether to display shape information.
@param show_dtype whether to display layer dtypes.
@param show_layer_names whether to display layer names.
@param rankdir `rankdir` argument passed to PyDot,
    a string specifying the format of the plot: `"TB"`
    creates a vertical plot; `"LR"` creates a horizontal plot.
@param expand_nested whether to expand nested Functional models
    into clusters.
@param dpi Image resolution in dots per inch.
@param subgraph whether to return a `pydot.Cluster` instance.
@param show_layer_activations Display layer activations (only for layers that
    have an `activation` property).
@param show_trainable whether to display if a layer is trainable.
@param ... Passed on to the Python callable

@export
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/model_to_dot>
