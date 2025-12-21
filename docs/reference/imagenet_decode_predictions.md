# Decodes the prediction of an ImageNet model.

Decodes the prediction of an ImageNet model.

## Usage

``` r
imagenet_decode_predictions(preds, top = 5)
```

## Arguments

- preds:

  Tensor encoding a batch of predictions.

- top:

  integer, how many top-guesses to return.

## Value

List of data frames with variables `class_name`, `class_description`,
and `score` (one data frame per sample in batch input).
