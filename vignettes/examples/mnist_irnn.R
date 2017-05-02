#' This is a reproduction of the IRNN experiment
#' with pixel-by-pixel sequential MNIST in
#' "A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"
#' by Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton
#' 
#' arxiv:1504.00941v2 [cs.NE] 7 Apr 2015
#' http://arxiv.org/pdf/1504.00941v2.pdf
# 
#' Optimizer is replaced with RMSprop which yields more stable and steady
#' improvement.
# 
#' Reaches 0.93 train/test accuracy after 900 epochs
#' (which roughly corresponds to 1687500 steps in the original paper.)

library(keras)
