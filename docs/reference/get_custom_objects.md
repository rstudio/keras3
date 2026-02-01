# Get/set the currently registered custom objects.

Custom objects set using `custom_object_scope()` are not added to the
global list of custom objects, and will not appear in the returned list.

## Usage

``` r
get_custom_objects()

set_custom_objects(objects = named_list(), clear = TRUE)
```

## Arguments

- objects:

  A named list of custom objects, as returned by `get_custom_objects()`
  and `set_custom_objects()`.

- clear:

  bool, whether to clear the custom object registry before populating it
  with `objects`.

## Value

An R named list mapping registered names to registered objects.
`set_custom_objects()` returns the registry values before updating,
invisibly.

## Note

[`register_keras_serializable()`](https://keras3.posit.co/reference/register_keras_serializable.md)
is preferred over `set_custom_objects()` for registering new objects.

## Examples

    get_custom_objects()

You can use `set_custom_objects()` to restore a previous registry state.

    # within a function, if you want to temporarily modify the registry,
    function() {
      orig_objects <- set_custom_objects(clear = TRUE)
      on.exit(set_custom_objects(orig_objects))

      ## temporarily modify the global registry
      # register_keras_serializable(....)
      # ....  <do work>
      # on.exit(), the previous registry state is restored.
    }

## See also

Other serialization utilities:  
[`deserialize_keras_object()`](https://keras3.posit.co/reference/deserialize_keras_object.md)  
[`get_registered_name()`](https://keras3.posit.co/reference/get_registered_name.md)  
[`get_registered_object()`](https://keras3.posit.co/reference/get_registered_object.md)  
[`register_keras_serializable()`](https://keras3.posit.co/reference/register_keras_serializable.md)  
[`serialize_keras_object()`](https://keras3.posit.co/reference/serialize_keras_object.md)  
[`with_custom_object_scope()`](https://keras3.posit.co/reference/with_custom_object_scope.md)  
