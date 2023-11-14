Pad `images` with zeros to the specified `height` and `width`.

@description

# Examples

```r
images <- random_uniform(c(15, 25, 3))
padded_images <- k_image_pad_images(
    images, 2, 3, target_height = 20, target_width = 30
)
shape(padded_images)
```

```
## (20, 30, 3)
```


```r
batch_images <- random_uniform(c(2, 15, 25, 3))
padded_batch <- k_image_pad_images(batch_images, 2, 3,
                                   target_height = 20,
                                   target_width = 30)
shape(padded_batch)
```

```
## (2, 20, 30, 3)
```

@returns
If `images` were 4D, a 4D float Tensor of shape
    `(batch, target_height, target_width, channels)`
If `images` were 3D, a 3D float Tensor of shape
    `(target_height, target_width, channels)`

@param images
4D Tensor of shape `(batch, height, width, channels)` or 3D
Tensor of shape `(height, width, channels)`.

@param top_padding
Number of rows of zeros to add on top.

@param bottom_padding
Number of rows of zeros to add at the bottom.

@param left_padding
Number of columns of zeros to add on the left.

@param right_padding
Number of columns of zeros to add on the right.

@param target_height
Height of output images.

@param target_width
Width of output images.

@export
@family image ops
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/image/pad_images>

