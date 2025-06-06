% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ops.R
\name{op_hard_silu}
\alias{op_hard_silu}
\alias{op_hard_swish}
\title{Hard SiLU activation function, also known as Hard Swish.}
\usage{
op_hard_silu(x)

op_hard_swish(x)
}
\arguments{
\item{x}{Input tensor.}
}
\value{
A tensor with the same shape as \code{x}.
}
\description{
It is defined as:
\itemize{
\item \code{0} if \verb{if x < -3}
\item \code{x} if \code{x > 3}
\item \code{x * (x + 3) / 6} if \verb{-3 <= x <= 3}
}

It's a faster, piecewise linear approximation of the silu activation.
}
\section{Examples}{
\if{html}{\out{<div class="sourceCode r">}}\preformatted{x <- op_convert_to_tensor(c(-3.0, -1.0, 0.0, 1.0, 3.0))
op_hard_silu(x)
}\if{html}{\out{</div>}}

\if{html}{\out{<div class="sourceCode">}}\preformatted{## tf.Tensor([-0.         -0.33333333  0.          0.66666667  3.        ], shape=(5), dtype=float64)

}\if{html}{\out{</div>}}
}

\seealso{
Other nn ops: \cr
\code{\link{op_average_pool}()} \cr
\code{\link{op_batch_normalization}()} \cr
\code{\link{op_binary_crossentropy}()} \cr
\code{\link{op_categorical_crossentropy}()} \cr
\code{\link{op_celu}()} \cr
\code{\link{op_conv}()} \cr
\code{\link{op_conv_transpose}()} \cr
\code{\link{op_ctc_loss}()} \cr
\code{\link{op_depthwise_conv}()} \cr
\code{\link{op_dot_product_attention}()} \cr
\code{\link{op_elu}()} \cr
\code{\link{op_gelu}()} \cr
\code{\link{op_glu}()} \cr
\code{\link{op_hard_shrink}()} \cr
\code{\link{op_hard_sigmoid}()} \cr
\code{\link{op_hard_tanh}()} \cr
\code{\link{op_leaky_relu}()} \cr
\code{\link{op_log_sigmoid}()} \cr
\code{\link{op_log_softmax}()} \cr
\code{\link{op_max_pool}()} \cr
\code{\link{op_moments}()} \cr
\code{\link{op_multi_hot}()} \cr
\code{\link{op_normalize}()} \cr
\code{\link{op_one_hot}()} \cr
\code{\link{op_polar}()} \cr
\code{\link{op_psnr}()} \cr
\code{\link{op_relu}()} \cr
\code{\link{op_relu6}()} \cr
\code{\link{op_rms_normalization}()} \cr
\code{\link{op_selu}()} \cr
\code{\link{op_separable_conv}()} \cr
\code{\link{op_sigmoid}()} \cr
\code{\link{op_silu}()} \cr
\code{\link{op_soft_shrink}()} \cr
\code{\link{op_softmax}()} \cr
\code{\link{op_softplus}()} \cr
\code{\link{op_softsign}()} \cr
\code{\link{op_sparse_categorical_crossentropy}()} \cr
\code{\link{op_sparse_plus}()} \cr
\code{\link{op_sparsemax}()} \cr
\code{\link{op_squareplus}()} \cr
\code{\link{op_tanh_shrink}()} \cr
\code{\link{op_threshold}()} \cr
\code{\link{op_unravel_index}()} \cr

Other ops: \cr
\code{\link{op_abs}()} \cr
\code{\link{op_add}()} \cr
\code{\link{op_all}()} \cr
\code{\link{op_any}()} \cr
\code{\link{op_append}()} \cr
\code{\link{op_arange}()} \cr
\code{\link{op_arccos}()} \cr
\code{\link{op_arccosh}()} \cr
\code{\link{op_arcsin}()} \cr
\code{\link{op_arcsinh}()} \cr
\code{\link{op_arctan}()} \cr
\code{\link{op_arctan2}()} \cr
\code{\link{op_arctanh}()} \cr
\code{\link{op_argmax}()} \cr
\code{\link{op_argmin}()} \cr
\code{\link{op_argpartition}()} \cr
\code{\link{op_argsort}()} \cr
\code{\link{op_array}()} \cr
\code{\link{op_associative_scan}()} \cr
\code{\link{op_average}()} \cr
\code{\link{op_average_pool}()} \cr
\code{\link{op_batch_normalization}()} \cr
\code{\link{op_binary_crossentropy}()} \cr
\code{\link{op_bincount}()} \cr
\code{\link{op_bitwise_and}()} \cr
\code{\link{op_bitwise_invert}()} \cr
\code{\link{op_bitwise_left_shift}()} \cr
\code{\link{op_bitwise_not}()} \cr
\code{\link{op_bitwise_or}()} \cr
\code{\link{op_bitwise_right_shift}()} \cr
\code{\link{op_bitwise_xor}()} \cr
\code{\link{op_broadcast_to}()} \cr
\code{\link{op_cast}()} \cr
\code{\link{op_categorical_crossentropy}()} \cr
\code{\link{op_ceil}()} \cr
\code{\link{op_celu}()} \cr
\code{\link{op_cholesky}()} \cr
\code{\link{op_clip}()} \cr
\code{\link{op_concatenate}()} \cr
\code{\link{op_cond}()} \cr
\code{\link{op_conj}()} \cr
\code{\link{op_conv}()} \cr
\code{\link{op_conv_transpose}()} \cr
\code{\link{op_convert_to_numpy}()} \cr
\code{\link{op_convert_to_tensor}()} \cr
\code{\link{op_copy}()} \cr
\code{\link{op_correlate}()} \cr
\code{\link{op_cos}()} \cr
\code{\link{op_cosh}()} \cr
\code{\link{op_count_nonzero}()} \cr
\code{\link{op_cross}()} \cr
\code{\link{op_ctc_decode}()} \cr
\code{\link{op_ctc_loss}()} \cr
\code{\link{op_cumprod}()} \cr
\code{\link{op_cumsum}()} \cr
\code{\link{op_custom_gradient}()} \cr
\code{\link{op_depthwise_conv}()} \cr
\code{\link{op_det}()} \cr
\code{\link{op_diag}()} \cr
\code{\link{op_diagflat}()} \cr
\code{\link{op_diagonal}()} \cr
\code{\link{op_diff}()} \cr
\code{\link{op_digitize}()} \cr
\code{\link{op_divide}()} \cr
\code{\link{op_divide_no_nan}()} \cr
\code{\link{op_dot}()} \cr
\code{\link{op_dot_product_attention}()} \cr
\code{\link{op_dtype}()} \cr
\code{\link{op_eig}()} \cr
\code{\link{op_eigh}()} \cr
\code{\link{op_einsum}()} \cr
\code{\link{op_elu}()} \cr
\code{\link{op_empty}()} \cr
\code{\link{op_equal}()} \cr
\code{\link{op_erf}()} \cr
\code{\link{op_erfinv}()} \cr
\code{\link{op_exp}()} \cr
\code{\link{op_exp2}()} \cr
\code{\link{op_expand_dims}()} \cr
\code{\link{op_expm1}()} \cr
\code{\link{op_extract_sequences}()} \cr
\code{\link{op_eye}()} \cr
\code{\link{op_fft}()} \cr
\code{\link{op_fft2}()} \cr
\code{\link{op_flip}()} \cr
\code{\link{op_floor}()} \cr
\code{\link{op_floor_divide}()} \cr
\code{\link{op_fori_loop}()} \cr
\code{\link{op_full}()} \cr
\code{\link{op_full_like}()} \cr
\code{\link{op_gelu}()} \cr
\code{\link{op_get_item}()} \cr
\code{\link{op_glu}()} \cr
\code{\link{op_greater}()} \cr
\code{\link{op_greater_equal}()} \cr
\code{\link{op_hard_shrink}()} \cr
\code{\link{op_hard_sigmoid}()} \cr
\code{\link{op_hard_tanh}()} \cr
\code{\link{op_histogram}()} \cr
\code{\link{op_hstack}()} \cr
\code{\link{op_identity}()} \cr
\code{\link{op_ifft2}()} \cr
\code{\link{op_imag}()} \cr
\code{\link{op_image_affine_transform}()} \cr
\code{\link{op_image_crop}()} \cr
\code{\link{op_image_extract_patches}()} \cr
\code{\link{op_image_gaussian_blur}()} \cr
\code{\link{op_image_hsv_to_rgb}()} \cr
\code{\link{op_image_map_coordinates}()} \cr
\code{\link{op_image_pad}()} \cr
\code{\link{op_image_perspective_transform}()} \cr
\code{\link{op_image_resize}()} \cr
\code{\link{op_image_rgb_to_grayscale}()} \cr
\code{\link{op_image_rgb_to_hsv}()} \cr
\code{\link{op_in_top_k}()} \cr
\code{\link{op_inner}()} \cr
\code{\link{op_inv}()} \cr
\code{\link{op_irfft}()} \cr
\code{\link{op_is_tensor}()} \cr
\code{\link{op_isclose}()} \cr
\code{\link{op_isfinite}()} \cr
\code{\link{op_isinf}()} \cr
\code{\link{op_isnan}()} \cr
\code{\link{op_istft}()} \cr
\code{\link{op_leaky_relu}()} \cr
\code{\link{op_left_shift}()} \cr
\code{\link{op_less}()} \cr
\code{\link{op_less_equal}()} \cr
\code{\link{op_linspace}()} \cr
\code{\link{op_log}()} \cr
\code{\link{op_log10}()} \cr
\code{\link{op_log1p}()} \cr
\code{\link{op_log2}()} \cr
\code{\link{op_log_sigmoid}()} \cr
\code{\link{op_log_softmax}()} \cr
\code{\link{op_logaddexp}()} \cr
\code{\link{op_logdet}()} \cr
\code{\link{op_logical_and}()} \cr
\code{\link{op_logical_not}()} \cr
\code{\link{op_logical_or}()} \cr
\code{\link{op_logical_xor}()} \cr
\code{\link{op_logspace}()} \cr
\code{\link{op_logsumexp}()} \cr
\code{\link{op_lstsq}()} \cr
\code{\link{op_lu_factor}()} \cr
\code{\link{op_map}()} \cr
\code{\link{op_matmul}()} \cr
\code{\link{op_max}()} \cr
\code{\link{op_max_pool}()} \cr
\code{\link{op_maximum}()} \cr
\code{\link{op_mean}()} \cr
\code{\link{op_median}()} \cr
\code{\link{op_meshgrid}()} \cr
\code{\link{op_min}()} \cr
\code{\link{op_minimum}()} \cr
\code{\link{op_mod}()} \cr
\code{\link{op_moments}()} \cr
\code{\link{op_moveaxis}()} \cr
\code{\link{op_multi_hot}()} \cr
\code{\link{op_multiply}()} \cr
\code{\link{op_nan_to_num}()} \cr
\code{\link{op_ndim}()} \cr
\code{\link{op_negative}()} \cr
\code{\link{op_nonzero}()} \cr
\code{\link{op_norm}()} \cr
\code{\link{op_normalize}()} \cr
\code{\link{op_not_equal}()} \cr
\code{\link{op_one_hot}()} \cr
\code{\link{op_ones}()} \cr
\code{\link{op_ones_like}()} \cr
\code{\link{op_outer}()} \cr
\code{\link{op_pad}()} \cr
\code{\link{op_polar}()} \cr
\code{\link{op_power}()} \cr
\code{\link{op_prod}()} \cr
\code{\link{op_psnr}()} \cr
\code{\link{op_qr}()} \cr
\code{\link{op_quantile}()} \cr
\code{\link{op_ravel}()} \cr
\code{\link{op_real}()} \cr
\code{\link{op_rearrange}()} \cr
\code{\link{op_reciprocal}()} \cr
\code{\link{op_relu}()} \cr
\code{\link{op_relu6}()} \cr
\code{\link{op_repeat}()} \cr
\code{\link{op_reshape}()} \cr
\code{\link{op_rfft}()} \cr
\code{\link{op_right_shift}()} \cr
\code{\link{op_rms_normalization}()} \cr
\code{\link{op_roll}()} \cr
\code{\link{op_rot90}()} \cr
\code{\link{op_round}()} \cr
\code{\link{op_rsqrt}()} \cr
\code{\link{op_saturate_cast}()} \cr
\code{\link{op_scan}()} \cr
\code{\link{op_scatter}()} \cr
\code{\link{op_scatter_update}()} \cr
\code{\link{op_searchsorted}()} \cr
\code{\link{op_segment_max}()} \cr
\code{\link{op_segment_sum}()} \cr
\code{\link{op_select}()} \cr
\code{\link{op_selu}()} \cr
\code{\link{op_separable_conv}()} \cr
\code{\link{op_shape}()} \cr
\code{\link{op_sigmoid}()} \cr
\code{\link{op_sign}()} \cr
\code{\link{op_signbit}()} \cr
\code{\link{op_silu}()} \cr
\code{\link{op_sin}()} \cr
\code{\link{op_sinh}()} \cr
\code{\link{op_size}()} \cr
\code{\link{op_slice}()} \cr
\code{\link{op_slice_update}()} \cr
\code{\link{op_slogdet}()} \cr
\code{\link{op_soft_shrink}()} \cr
\code{\link{op_softmax}()} \cr
\code{\link{op_softplus}()} \cr
\code{\link{op_softsign}()} \cr
\code{\link{op_solve}()} \cr
\code{\link{op_solve_triangular}()} \cr
\code{\link{op_sort}()} \cr
\code{\link{op_sparse_categorical_crossentropy}()} \cr
\code{\link{op_sparse_plus}()} \cr
\code{\link{op_sparsemax}()} \cr
\code{\link{op_split}()} \cr
\code{\link{op_sqrt}()} \cr
\code{\link{op_square}()} \cr
\code{\link{op_squareplus}()} \cr
\code{\link{op_squeeze}()} \cr
\code{\link{op_stack}()} \cr
\code{\link{op_std}()} \cr
\code{\link{op_stft}()} \cr
\code{\link{op_stop_gradient}()} \cr
\code{\link{op_subset}()} \cr
\code{\link{op_subtract}()} \cr
\code{\link{op_sum}()} \cr
\code{\link{op_svd}()} \cr
\code{\link{op_swapaxes}()} \cr
\code{\link{op_switch}()} \cr
\code{\link{op_take}()} \cr
\code{\link{op_take_along_axis}()} \cr
\code{\link{op_tan}()} \cr
\code{\link{op_tanh}()} \cr
\code{\link{op_tanh_shrink}()} \cr
\code{\link{op_tensordot}()} \cr
\code{\link{op_threshold}()} \cr
\code{\link{op_tile}()} \cr
\code{\link{op_top_k}()} \cr
\code{\link{op_trace}()} \cr
\code{\link{op_transpose}()} \cr
\code{\link{op_tri}()} \cr
\code{\link{op_tril}()} \cr
\code{\link{op_triu}()} \cr
\code{\link{op_trunc}()} \cr
\code{\link{op_unravel_index}()} \cr
\code{\link{op_unstack}()} \cr
\code{\link{op_var}()} \cr
\code{\link{op_vdot}()} \cr
\code{\link{op_vectorize}()} \cr
\code{\link{op_vectorized_map}()} \cr
\code{\link{op_vstack}()} \cr
\code{\link{op_where}()} \cr
\code{\link{op_while_loop}()} \cr
\code{\link{op_zeros}()} \cr
\code{\link{op_zeros_like}()} \cr
}
\concept{nn ops}
\concept{ops}
