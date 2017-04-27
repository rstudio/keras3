# Trains two recurrent neural networks based upon a story and a question.
# The resulting merged vector is then queried to answer a range of bAbI tasks.
# 
# The results are comparable to those for an LSTM model provided in Weston et al.:
# "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
# http://arxiv.org/abs/1502.05698
# 
# Task Number                  | FB LSTM Baseline | Keras QA
# ---                          | ---              | ---
# QA1 - Single Supporting Fact | 50               | 100.0
# QA2 - Two Supporting Facts   | 20               | 50.0
# QA3 - Three Supporting Facts | 20               | 20.5
# QA4 - Two Arg. Relations     | 61               | 62.9
# QA5 - Three Arg. Relations   | 70               | 61.9
# QA6 - yes/No Questions       | 48               | 50.7
# QA7 - Counting               | 49               | 78.9
# QA8 - Lists/Sets             | 45               | 77.2
# QA9 - Simple Negation        | 64               | 64.0
# QA10 - Indefinite Knowledge  | 44               | 47.7
# QA11 - Basic Coreference     | 72               | 74.9
# QA12 - Conjunction           | 74               | 76.4
# QA13 - Compound Coreference  | 94               | 94.4
# QA14 - Time Reasoning        | 27               | 34.8
# QA15 - Basic Deduction       | 21               | 32.4
# QA16 - Basic Induction       | 23               | 50.6
# QA17 - Positional Reasoning  | 51               | 49.1
# QA18 - Size Reasoning        | 52               | 90.8
# QA19 - Path Finding          | 8                | 9.0
# QA20 - Agent's Motivations   | 91               | 90.7
# 
# For the resources related to the bAbI project, refer to:
#   https://research.facebook.com/researchers/1543934539189348
# 
# Notes:
#   
#   - With default word, sentence, and query vector sizes, the GRU model achieves:
#   - 100% test accuracy on QA1 in 20 epochs (2 seconds per epoch on CPU)
# - 50% test accuracy on QA2 in 20 epochs (16 seconds per epoch on CPU)
# In comparison, the Facebook paper achieves 50% and 20% for the LSTM baseline.
# 
# - The task does not traditionally parse the question separately. This likely
# improves accuracy and is a good example of merging two RNNs.
# 
# - The word vector embeddings are not shared between the story and question RNNs.
# 
# - See how the accuracy changes given 10,000 training samples (en-10k) instead
# of only 1000. 1000 was used in order to be comparable to the original paper.
# 
# - Experiment with GRU, LSTM, and JZS1-3 as they give subtly different results.
# 
# - The length and noise (i.e. 'useless' story components) impact the ability for
# LSTMs / GRUs to provide the correct answer. Given only the supporting facts,
# these RNNs can achieve 100% accuracy on many tasks. Memory networks and neural
# networks that use attentional processes can efficiently search through this
# noise to find the relevant statements, improving performance substantially.
# This becomes especially obvious on QA2 and QA3, both far longer than QA1.
# 

library(keras)
library(readr)
library(stringr)
library(purrr)
library(tibble)
library(dplyr)


# Function definition -----------------------------------------------------

tokenize_words <- function(x){
  x %>% 
    str_replace_all('([[:punct:]]+)', ' \\1') %>% 
    str_split(' ') %>%
    unlist()
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
    mutate(story = map(story, ~tokenize_words(.x))) %>%
    ungroup() %>%
    mutate(id = row_number()) %>%
    select(id, question, answer, story)
}

# Parameters --------------------------------------------------------------

max_length <- 99999

# Data Preparation --------------------------------------------------------

download.file(
  url = "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz", 
  destfile = "babi-tasks-v1-2.tar.gz"
  )

untar("babi-tasks-v1-2.tar.gz")

# Default QA1 with 1000 samples
# challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
# QA1 with 10,000 samples
# challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
# QA2 with 1000 samples
challenge <- "tasks_1-20_v1-2/en/qa2_two-supporting-facts_%s.txt"
# QA2 with 10,000 samples
# challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'

train <- read_lines(sprintf(challenge, "train")) %>%
  parse_stories() %>%
  filter(map_int(story, ~length(.x)) <= max_length)

test <- read_lines(sprintf(challenge, "test")) %>%
  parse_stories() %>%
  filter(map_int(story, ~length(.x)) <= max_length)
