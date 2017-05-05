#' Trains a LSTM on the IMDB sentiment classification task.
#' 
#' The dataset is actually too small for LSTM to be of any advantage compared to
#' simpler, much faster methods such as TF-IDF + LogReg.
#' 
#' Notes:
#' 
#' - RNNs are tricky. Choice of batch size is important, choice of loss and
#' optimizer is critical, etc. Some configurations won't converge.
#' 
#' - LSTM loss decrease patterns during training can be quite different from
#' what you see with CNNs/MLPs/etc.

library(keras)
