#' This script loads pre-trained word embeddings (GloVe embeddings) into a
#' frozen Keras Embedding layer, and uses it to train a text classification
#' model on the 20 Newsgroup dataset (classication of newsgroup messages into 20
#' different categories).
#' 
#' GloVe embedding data can be found at: 
#' http://nlp.stanford.edu/data/glove.6B.zip (source page:
#' http://nlp.stanford.edu/projects/glove/)
#'
#' 20 Newsgroup data can be found at: 
#' http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
#' 

library(keras)
