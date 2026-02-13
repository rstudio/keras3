# Zip lists

This is conceptually similar to
[`zip()`](https://rdrr.io/r/utils/zip.html) in Python, or R functions
[`purrr::transpose()`](https://purrr.tidyverse.org/reference/transpose.html)
and
[`data.table::transpose()`](https://rdrr.io/pkg/data.table/man/transpose.html)
(albeit, accepting elements in `...` instead of a single list), with one
crucial difference: if the provided objects are named, then matching is
done by names, not positions.

## Usage

``` r
zip_lists(...)
```

## Arguments

- ...:

  R lists or atomic vectors, optionally named.

## Value

A inverted list

## Details

All arguments supplied must be of the same length. If positional
matching is required, then all arguments provided must be unnamed. If
matching by names, then all arguments must have the same set of names,
but they can be in different orders.

## See also

Other utils:  
[`audio_dataset_from_directory()`](https://keras3.posit.co/reference/audio_dataset_from_directory.md)  
[`clear_session()`](https://keras3.posit.co/reference/clear_session.md)  
[`config_disable_interactive_logging()`](https://keras3.posit.co/reference/config_disable_interactive_logging.md)  
[`config_disable_traceback_filtering()`](https://keras3.posit.co/reference/config_disable_traceback_filtering.md)  
[`config_enable_interactive_logging()`](https://keras3.posit.co/reference/config_enable_interactive_logging.md)  
[`config_enable_traceback_filtering()`](https://keras3.posit.co/reference/config_enable_traceback_filtering.md)  
[`config_is_interactive_logging_enabled()`](https://keras3.posit.co/reference/config_is_interactive_logging_enabled.md)  
[`config_is_traceback_filtering_enabled()`](https://keras3.posit.co/reference/config_is_traceback_filtering_enabled.md)  
[`get_file()`](https://keras3.posit.co/reference/get_file.md)  
[`get_source_inputs()`](https://keras3.posit.co/reference/get_source_inputs.md)  
[`image_array_save()`](https://keras3.posit.co/reference/image_array_save.md)  
[`image_dataset_from_directory()`](https://keras3.posit.co/reference/image_dataset_from_directory.md)  
[`image_from_array()`](https://keras3.posit.co/reference/image_from_array.md)  
[`image_load()`](https://keras3.posit.co/reference/image_load.md)  
[`image_smart_resize()`](https://keras3.posit.co/reference/image_smart_resize.md)  
[`image_to_array()`](https://keras3.posit.co/reference/image_to_array.md)  
[`layer_feature_space()`](https://keras3.posit.co/reference/layer_feature_space.md)  
[`normalize()`](https://keras3.posit.co/reference/normalize.md)  
[`pad_sequences()`](https://keras3.posit.co/reference/pad_sequences.md)  
[`set_random_seed()`](https://keras3.posit.co/reference/set_random_seed.md)  
[`split_dataset()`](https://keras3.posit.co/reference/split_dataset.md)  
[`text_dataset_from_directory()`](https://keras3.posit.co/reference/text_dataset_from_directory.md)  
[`timeseries_dataset_from_array()`](https://keras3.posit.co/reference/timeseries_dataset_from_array.md)  
[`to_categorical()`](https://keras3.posit.co/reference/to_categorical.md)  

## Examples

``` r
gradients <- list("grad_for_wt_1", "grad_for_wt_2", "grad_for_wt_3")
weights <- list("weight_1", "weight_2", "weight_3")
str(zip_lists(gradients, weights))
#> List of 3
#>  $ :List of 2
#>   ..$ : chr "grad_for_wt_1"
#>   ..$ : chr "weight_1"
#>  $ :List of 2
#>   ..$ : chr "grad_for_wt_2"
#>   ..$ : chr "weight_2"
#>  $ :List of 2
#>   ..$ : chr "grad_for_wt_3"
#>   ..$ : chr "weight_3"
str(zip_lists(gradient = gradients, weight = weights))
#> List of 3
#>  $ :List of 2
#>   ..$ gradient: chr "grad_for_wt_1"
#>   ..$ weight  : chr "weight_1"
#>  $ :List of 2
#>   ..$ gradient: chr "grad_for_wt_2"
#>   ..$ weight  : chr "weight_2"
#>  $ :List of 2
#>   ..$ gradient: chr "grad_for_wt_3"
#>   ..$ weight  : chr "weight_3"

names(gradients) <- names(weights) <- paste0("layer_", 1:3)
str(zip_lists(gradients, weights[c(3, 1, 2)]))
#> List of 3
#>  $ layer_1:List of 2
#>   ..$ : chr "grad_for_wt_1"
#>   ..$ : chr "weight_1"
#>  $ layer_2:List of 2
#>   ..$ : chr "grad_for_wt_2"
#>   ..$ : chr "weight_2"
#>  $ layer_3:List of 2
#>   ..$ : chr "grad_for_wt_3"
#>   ..$ : chr "weight_3"

names(gradients) <- paste0("gradient_", 1:3)
try(zip_lists(gradients, weights)) # error, names don't match
#> Error in zip_lists(gradients, weights) : 
#>   All names of arguments provided to `zip_lists()` must match. Call `unname()` on each argument if you want positional matching
# call unname directly for positional matching
str(zip_lists(unname(gradients), unname(weights)))
#> List of 3
#>  $ :List of 2
#>   ..$ : chr "grad_for_wt_1"
#>   ..$ : chr "weight_1"
#>  $ :List of 2
#>   ..$ : chr "grad_for_wt_2"
#>   ..$ : chr "weight_2"
#>  $ :List of 2
#>   ..$ : chr "grad_for_wt_3"
#>   ..$ : chr "weight_3"
```
