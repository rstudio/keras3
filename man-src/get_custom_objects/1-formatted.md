Retrieves a live reference to the global dictionary of custom objects.

@description
Custom objects set using using `custom_object_scope()` are not added to the
global dictionary of custom objects, and will not appear in the returned
dictionary.

# Examples
```python
get_custom_objects().clear()
get_custom_objects()['MyObject'] = MyObject
```

@returns
    Global dictionary mapping registered class names to classes.

@export
@family registration object saving
@family object saving
@family saving
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_custom_objects>
