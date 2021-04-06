from models import MLE, HMM
from sentences_parser import load_sentences_data, adapt_data_by_words
import numpy as np

train_sentences, test_sentences = load_sentences_data()

# Baseline model
train_words, test_words = adapt_data_by_words(train_sentences, test_sentences)
print("Baseline model:\n")
baseline = MLE()
baseline.fit(train_words)
baseline.evaluate(test_words)

# Base HMM
print("Base HMM:")
hmm_base = HMM()
hmm_base.fit(train_sentences)
hmm_base.evaluate(test_sentences)

# # HMM with Add-1 smoothing
print("HMM with Add-1 smoothing:")
hmm_smooth = HMM(laplace_smooth_factor=1)
hmm_smooth.fit(train_sentences)
hmm_smooth.evaluate(test_sentences)

print("HMM with Pseudo words:")
hmm_pseudo = HMM(create_pseudo_words_threshold=2)
hmm_pseudo.fit(train_sentences)
hmm_pseudo.evaluate(test_sentences)

print("HMM with Pseudo words and Add-1 smoothing:")
hmm_pseudo_smooth = HMM(laplace_smooth_factor=1, create_pseudo_words_threshold=2)
hmm_pseudo_smooth.fit(train_sentences)
evaluation_df = hmm_pseudo_smooth.evaluate(test_sentences)
evaluation_df.to_csv("evaluation.csv")
hmm_pseudo_smooth.confusion_matrix(evaluation_df)





