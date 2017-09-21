#' Trains a memory network on the bAbI dataset.
#' 
#' References:
#' 
#' - Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
#'   "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
#'   http://arxiv.org/abs/1502.05698
#' 
#' - Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
#'   "End-To-End Memory Networks", http://arxiv.org/abs/1503.08895
#' 
#' Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs.
#' Time per epoch: 3s on CPU (core i7).
#' 

library(keras)
library(readr)
library(stringr)
library(purrr)
library(tibble)
library(dplyr)

# Function definition -----------------------------------------------------

tokenize_words <- function(x){
  x <- x %>% 
    str_replace_all('([[:punct:]]+)', ' \\1') %>% 
    str_split(' ') %>%
    unlist()
  x[x != ""]
}

parse_stories <- function(lines, only_supporting = FALSE){
  lines <- lines %>% 
    str_split(" ", n = 2) %>%
    map_df(~tibble(nid = as.integer(.x[[1]]), line = .x[[2]]))
  
  lines <- lines %>%
    mutate(
      split = map(line, ~str_split(.x, "\t")[[1]]),
      q = map_chr(split, ~.x[1]),
      a = map_chr(split, ~.x[2]),
      supporting = map(split, ~.x[3] %>% str_split(" ") %>% unlist() %>% as.integer()),
      story_id = c(0, cumsum(nid[-nrow(.)] > nid[-1]))
    ) %>%
    select(-split)
  
  stories <- lines %>%
    filter(is.na(a)) %>%
    select(nid_story = nid, story_id, story = q)
  
  questions <- lines %>%
    filter(!is.na(a)) %>%
    select(-line) %>%
    left_join(stories, by = "story_id") %>%
    filter(nid_story < nid)
  
  if(only_supporting){
    questions <- questions %>%
      filter(map2_lgl(nid_story, supporting, ~.x %in% .y))
  }
  
  questions %>%
    group_by(story_id, nid, question = q, answer = a) %>%
    summarise(story = paste(story, collapse = " ")) %>%
    ungroup() %>% 
    mutate(
      question = map(question, ~tokenize_words(.x)),
      story = map(story, ~tokenize_words(.x)),
      id = row_number()
    ) %>%
    select(id, question, answer, story)
}

vectorize_stories <- function(data, vocab, story_maxlen, query_maxlen){
  
  questions <- map(data$question, function(x){
    map_int(x, ~which(.x == vocab))
  })
  
  stories <- map(data$story, function(x){
    map_int(x, ~which(.x == vocab))
  })
  
  # "" represents padding
  answers <- sapply(c("", vocab), function(x){
    as.integer(x == data$answer)
  })
  
  list(
    questions = pad_sequences(questions, maxlen = query_maxlen),
    stories   = pad_sequences(stories, maxlen = story_maxlen),
    answers   = answers
  )
}


# Parameters --------------------------------------------------------------

challenges <- list(
  # QA1 with 10,000 samples
  single_supporting_fact_10k = "%stasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_%s.txt",
  # QA2 with 10,000 samples
  two_supporting_facts_10k = "%stasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_%s.txt"
)

challenge_type <- "single_supporting_fact_10k"
challenge <- challenges[[challenge_type]]
max_length <- 999999


# Data Preparation --------------------------------------------------------

# Download data
path <- get_file(
  fname = "babi-tasks-v1-2.tar.gz",
  origin = "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz"
)
untar(path, exdir = str_replace(path, fixed(".tar.gz"), "/"))
path <- str_replace(path, fixed(".tar.gz"), "/")

# Reading training and test data
train <- read_lines(sprintf(challenge, path, "train")) %>%
  parse_stories() %>%
  filter(map_int(story, ~length(.x)) <= max_length)

test <- read_lines(sprintf(challenge, path, "test")) %>%
  parse_stories() %>%
  filter(map_int(story, ~length(.x)) <= max_length)

# Extract the vocabulary
all_data <- bind_rows(train, test)
vocab <- c(unlist(all_data$question), all_data$answer, 
           unlist(all_data$story)) %>%
  unique() %>%
  sort()

# Reserve 0 for masking via pad_sequences
vocab_size <- length(vocab) + 1
story_maxlen <- map_int(all_data$story, ~length(.x)) %>% max()
query_maxlen <- map_int(all_data$question, ~length(.x)) %>% max()

# Vectorized versions of training and test sets
train_vec <- vectorize_stories(train, vocab, story_maxlen, query_maxlen)
test_vec <- vectorize_stories(test, vocab, story_maxlen, query_maxlen)


# Defining the model ------------------------------------------------------

# Placeholders
sequence <- layer_input(shape = c(story_maxlen))
question <- layer_input(shape = c(query_maxlen))

# Encoders
# Embed the input sequence into a sequence of vectors
sequence_encoder_m <- keras_model_sequential()
sequence_encoder_m %>%
  layer_embedding(input_dim = vocab_size, output_dim = 64) %>%
  layer_dropout(rate = 0.3)
# output: (samples, story_maxlen, embedding_dim)

# Embed the input into a sequence of vectors of size query_maxlen
sequence_encoder_c <- keras_model_sequential()
sequence_encoder_c %>%
  layer_embedding(input_dim = vocab_size, output = query_maxlen) %>%
  layer_dropout(rate = 0.3)
# output: (samples, story_maxlen, query_maxlen)

# Embed the question into a sequence of vectors
question_encoder <- keras_model_sequential()
question_encoder %>%
  layer_embedding(input_dim = vocab_size, output_dim = 64, 
                  input_length = query_maxlen) %>%
  layer_dropout(rate = 0.3)
# output: (samples, query_maxlen, embedding_dim)

# Encode input sequence and questions (which are indices)
# to sequences of dense vectors
sequence_encoded_m <- sequence_encoder_m(sequence)
sequence_encoded_c <- sequence_encoder_c(sequence)
question_encoded <- question_encoder(question)

# Compute a 'match' between the first input vector sequence
# and the question vector sequence
# shape: `(samples, story_maxlen, query_maxlen)`
match <- list(sequence_encoded_m, question_encoded) %>%
  layer_dot(axes = c(2,2)) %>%
  layer_activation("softmax")

# Add the match matrix with the second input vector sequence
response <- list(match, sequence_encoded_c) %>%
  layer_add() %>%
  layer_permute(c(2,1))

# Concatenate the match matrix with the question vector sequence
answer <- list(response, question_encoded) %>%
  layer_concatenate() %>%
  # The original paper uses a matrix multiplication for this reduction step.
  # We choose to use an RNN instead.
  layer_lstm(32) %>%
  # One regularization layer -- more would probably be needed.
  layer_dropout(rate = 0.3) %>%
  layer_dense(vocab_size) %>%
  # We output a probability distribution over the vocabulary
  layer_activation("softmax")

# Build the final model
model <- keras_model(inputs = list(sequence, question), answer)
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)


# Training ----------------------------------------------------------------

model %>% fit(
  x = list(train_vec$stories, train_vec$questions),
  y = train_vec$answers,
  batch_size = 32,
  epochs = 120,
  validation_data = list(list(test_vec$stories, test_vec$questions), test_vec$answers)
)
