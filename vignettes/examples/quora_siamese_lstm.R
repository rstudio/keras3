#' In this tutorial we will use Keras to classify duplicated questions from Quora.
#' The dataset first appeared in the Kaggle competition 
#' [*Quora Question Pairs*](https://www.kaggle.com/c/quora-question-pairs).
#' The dataset consists of ~400k pairs of questions and a column indicating 
#' if the question pair is duplicated. 
#' 
#' Our implementation is inspired by the Siamese Recurrent Architecture, Mueller et al. [*Siamese recurrent architectures for learning sentence similarity*](https://dl.acm.org/citation.cfm?id=3016291), with small modifications like the similarity
#' measure and the embedding layers (The original paper uses pre-trained word vectors). Using this kind
#' of architecture dates back to 2005 with [Le Cun et al](https://dl.acm.org/citation.cfm?id=1068961) and is usefull for
#' verification tasks. The idea is to learn a function that maps input patterns into a
#' target space such that a similarity measure in the target space approximates
#' the “semantic” distance in the input space. 
#' 
#' After the competition, Quora also described their approach to this problem in 
#' this [blog post](https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning).
#' 

library(readr)
library(keras)
library(purrr)

FLAGS <- flags(
  flag_integer("vocab_size", 50000),
  flag_integer("max_len_padding", 20),
  flag_integer("embedding_size", 256),
  flag_numeric("regularization", 0.0001),
  flag_integer("seq_embedding_size", 512)
)

# Downloading Data --------------------------------------------------------

quora_data <- get_file(
  "quora_duplicate_questions.tsv",
  "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
)


# Pre-processing ----------------------------------------------------------

df <- read_tsv(quora_data)

tokenizer <- text_tokenizer(num_words = FLAGS$vocab_size)
fit_text_tokenizer(tokenizer, x = c(df$question1, df$question2))

question1 <- texts_to_sequences(tokenizer, df$question1)
question2 <- texts_to_sequences(tokenizer, df$question2)

question1 <- pad_sequences(question1, maxlen = FLAGS$max_len_padding, value = FLAGS$vocab_size + 1)
question2 <- pad_sequences(question2, maxlen = FLAGS$max_len_padding, value = FLAGS$vocab_size + 1)


# Model Definition --------------------------------------------------------

input1 <- layer_input(shape = c(FLAGS$max_len_padding))
input2 <- layer_input(shape = c(FLAGS$max_len_padding))

embedding <- layer_embedding(
  input_dim = FLAGS$vocab_size + 2, 
  output_dim = FLAGS$embedding_size, 
  input_length = FLAGS$max_len_padding, 
  embeddings_regularizer = regularizer_l2(l = FLAGS$regularization)
)
seq_emb <- layer_lstm(
  units = FLAGS$seq_embedding_size, 
  recurrent_regularizer = regularizer_l2(l = FLAGS$regularization)
)

vector1 <- embedding(input1) %>%
  seq_emb()
vector2 <- embedding(input2) %>%
  seq_emb()

out <- layer_dot(list(vector1, vector2), axes = 1) %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(list(input1, input2), out)
model %>% compile(
  optimizer = "adam", 
  loss = "binary_crossentropy", 
  metrics = list(
    acc = metric_binary_accuracy
  )
)

# Model Fitting -----------------------------------------------------------

set.seed(1817328)
val_sample <- sample.int(nrow(question1), size = 0.1*nrow(question1))

model %>%
  fit(
    list(question1[-val_sample,], question2[-val_sample,]),
    df$is_duplicate[-val_sample], 
    batch_size = 128, 
    epochs = 30, 
    validation_data = list(
      list(question1[val_sample,], question2[val_sample,]), df$is_duplicate[val_sample]
    ),
    callbacks = list(
      callback_early_stopping(patience = 5),
      callback_reduce_lr_on_plateau(patience = 3)
    )
  )

save_model_hdf5(model, "model-question-pairs.hdf5", include_optimizer = TRUE)
save_text_tokenizer(tokenizer, "tokenizer-question-pairs.hdf5")


# Prediction --------------------------------------------------------------
# In a fresh R session:
# Load model and tokenizer -

model <- load_model_hdf5("model-question-pairs.hdf5", compile = FALSE)
tokenizer <- load_text_tokenizer("tokenizer-question-pairs.hdf5")


predict_question_pairs <- function(model, tokenizer, q1, q2) {
  
  q1 <- texts_to_sequences(tokenizer, list(q1))
  q2 <- texts_to_sequences(tokenizer, list(q2))
  
  q1 <- pad_sequences(q1, 20)
  q2 <- pad_sequences(q2, 20)
  
  as.numeric(predict(model, list(q1, q2)))
}

# Getting predictions

predict_question_pairs(
  model, tokenizer, 
  q1 = "What is the main benefit of Quora?",
  q2 = "What are the advantages of using Quora?"
)

