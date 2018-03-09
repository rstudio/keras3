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

#'
#' IMPORTANT NOTE: This example does yet work correctly. The code executes fine and
#' appears to mimic the Python code upon which it is based however it achieves only
#' half the training accuracy that the Python code does so there is clearly a 
#' subtle difference.
#' 
#' We need to investigate this further before formally adding to the list of examples
#'
#'  

library(keras)

GLOVE_DIR <- 'glove.6B'
TEXT_DATA_DIR <- '20_newsgroup'
MAX_SEQUENCE_LENGTH <- 1000
MAX_NUM_WORDS <- 20000
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

texts <- character()  # text samples
labels <- integer() # label ids
labels_index <- list()  # dictionary: label name to numeric id

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

cat(sprintf('Found %s texts.\n', length(texts)))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer <- text_tokenizer(num_words=MAX_NUM_WORDS)
tokenizer %>% fit_text_tokenizer(texts)

# save the tokenizer in case we want to use it again
# for prediction within another R session, see:
# https://keras.rstudio.com/reference/save_text_tokenizer.html
save_text_tokenizer(tokenizer, "tokenizer")

sequences <- texts_to_sequences(tokenizer, texts)

word_index <- tokenizer$word_index
cat(sprintf('Found %s unique tokens.\n', length(word_index)))

data <- pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels <- to_categorical(labels)

cat('Shape of data tensor: ', dim(data), '\n')
cat('Shape of label tensor: ', dim(labels), '\n')

# split the data into a training set and a validation set
indices <- 1:nrow(data)
indices <- sample(indices)
data <- data[indices,]
labels <- labels[indices,]
num_validation_samples <- as.integer(VALIDATION_SPLIT * nrow(data))

x_train <- data[-(1:num_validation_samples),]
y_train <- labels[-(1:num_validation_samples),]
x_val <- data[1:num_validation_samples,]
y_val <- labels[1:num_validation_samples,]

cat('Preparing embedding matrix.\n')

# prepare embedding matrix
num_words <- min(MAX_NUM_WORDS, length(word_index) + 1)
prepare_embedding_matrix <- function() {
  embedding_matrix <- matrix(0L, nrow = num_words, ncol = EMBEDDING_DIM)
  for (word in names(word_index)) {
    index <- word_index[[word]]
    if (index >= MAX_NUM_WORDS)
      next
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector)) {
      # words not found in embedding index will be all-zeros.
      embedding_matrix[index,] <- embedding_vector
    }
  }
  embedding_matrix
}

embedding_matrix <- prepare_embedding_matrix()

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer <- layer_embedding(
  input_dim = num_words,
  output_dim = EMBEDDING_DIM,
  weights = list(embedding_matrix),
  input_length = MAX_SEQUENCE_LENGTH,
  trainable = FALSE
)
                           
cat('Training model\n')

# train a 1D convnet with global maxpooling
sequence_input <- layer_input(shape = list(MAX_SEQUENCE_LENGTH), dtype='int32')

preds <- sequence_input %>%
  embedding_layer %>% 
  layer_conv_1d(filters = 128, kernel_size = 5, activation = 'relu') %>% 
  layer_max_pooling_1d(pool_size = 5) %>% 
  layer_conv_1d(filters = 128, kernel_size = 5, activation = 'relu') %>% 
  layer_max_pooling_1d(pool_size = 5) %>% 
  layer_conv_1d(filters = 128, kernel_size = 5, activation = 'relu') %>% 
  layer_max_pooling_1d(pool_size = 35) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dense(units = length(labels_index), activation = 'softmax')


model <- keras_model(sequence_input, preds)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'rmsprop',
  metrics = c('acc')  
)

model %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 10,
  validation_data = list(x_val, y_val)
)




