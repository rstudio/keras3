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

# first, build index mapping words in the embeddings set
# to their embedding vector

cat('Indexing word vectors.\n')

embeddings_index <- new.env(parent = emptyenv())
lines <- readLines(file.path(GLOVE_DIR, 'glove.6B.100d.txt'))
for (line in lines) {
  values <- strsplit(line, ' ', fixed = TRUE)[[1]]
  word <- values[[1]]
  coefs <- as.numeric(values[-1])
  embeddings_index[[word]] <- coefs
}

cat(sprintf('Found %s word vectors.\n', length(embeddings_index)))

# second, prepare text samples and their labels
cat('Processing text dataset\n')

texts <- character()  # list of text samples
labels_index <- list()  # dictionary: label name to numeric id
labels <- character() # list of label ids

for (name in list.files(TEXT_DATA_DIR)) {
  path <- file.path(TEXT_DATA_DIR, name)
  if (file_test("-d", path)) {
    label_id <- length(labels_index)
    labels_index[[name]] <- label_id
    for (fname in list.files(path)) {
      if (grepl("^[0-9]+$", fname)) {
        fpath <- file.path(path, fname)
        t <- readLines(fpath, encoding = "latin1")
        t <- paste(t, collapse = "\n")
        i <- regexpr(pattern = '\n\n', t, fixed = TRUE)[[1]]
        if (i != -1L)
          t <- substring(t, i)
        texts <- c(texts, t)
        labels <- c(labels, label_id)
      }
    }
  }
}

cat(sprintf('Found %s texts.', length(texts)))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer <- text_tokenizer(num_words=MAX_NB_WORDS)
tokenizer %>% fit(texts)
sequences <- texts_to_sequences(tokenizer, texts)

word_index <- tokenizer$word_index
cat(sprintf('Found %s unique tokens.', length(word_index)))


# data <- pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# 
# labels = to_categorical(np.asarray(labels))
# print('Shape of data tensor:', data.shape)
# print('Shape of label tensor:', labels.shape)


