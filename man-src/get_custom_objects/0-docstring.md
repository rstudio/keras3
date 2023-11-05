Retrieves a live reference to the global dictionary of custom objects.

Custom objects set using using `custom_object_scope()` are not added to the
global dictionary of custom objects, and will not appear in the returned
dictionary.

Example:

```python
get_custom_objects().clear()
get_custom_objects()['MyObject'] = MyObject
```

Returns:
    Global dictionary mapping registered class names to classes.
