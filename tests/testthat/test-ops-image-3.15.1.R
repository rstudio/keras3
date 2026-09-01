test_that("new image ops expose 3D patches, transforms, edges, and SSIM", {
  skip_if_no_keras("3.15.1")

  volume <- op_ones(c(1, 4, 4, 4, 1))
  patches <- op_image_extract_patches_3d(volume, size = c(2, 2, 2))
  expect_tensor(patches, shape = c(1L, 2L, 2L, 2L, 8L))

  image <- op_reshape(op_arange(9, dtype = "float32"), c(3, 3))
  transformed <- op_image_scale_and_translate(
    image,
    output_shape = c(3, 3),
    scale = c(1, 1),
    translation = c(0, 0),
    spatial_dims = c(1, 2),
    method = "linear"
  )
  expect_equal(as.array(transformed), as.array(image), tolerance = 1e-6)

  images <- op_ones(c(1, 8, 8, 1))
  expect_tensor(
    op_image_sobel_edges(images),
    shape = c(1L, 8L, 8L, 1L, 2L)
  )
  expect_equal(
    as.numeric(op_image_ssim(images, images, filter_size = 3)),
    1,
    tolerance = 1e-6
  )
})
