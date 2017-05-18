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

GLOVE_DIR <- 'glove.6B'
TEXT_DATA_DIR <- '20_newsgroup'
MAX_SEQUENCE_LENGTH <- 1000
MAX_NB_WORDS <- 20000
EMBEDDING_DIM <- 100
VALIDATION_SPLIT <- 0.2

# download data if necessary
download_data <- function(data_dir, url_path, data_file) {
  if (!dir.exists(data_dir)) {
    download.file(paste0(url_path, data_file), data_file, mode = "wb")
    if (tools::file_ext(data_file) == "zip")
      unzip(data_file, exdir = tools::file_path_sans_ext(data_file))
    else
      untar(data_file)
    unlink(data_file)
  }
}
download_data(GLOVE_DIR, 'http://nlp.stanford.edu/data/', 'glove.6B.zip')
download_data(TEXT_DATA_DIR, "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/", "news20.tar.gz")





