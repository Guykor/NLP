import numpy as np
import pandas as pd

from sentences_parser import START_TOKEN, STOP_TOKEN


# inference over sentences
class Viterby:
    def __init__(self, transitions, emissions, unknown_words_default):
        self.S = transitions.reset_index().tag.unique()

        # making transition a table containing the cartesian product of tag (y_i-1) and
        # next_tag (y_i)
        emissions = emissions.unstack().fillna(1e-20).stack()
        transitions = transitions.unstack().fillna(1e-20)
        transitions.columns = transitions.columns.droplevel()

        self.__q_prob = transitions.to_numpy()
        self.__e_prob = emissions
        self.__start_token_idx = np.where(self.S == START_TOKEN)[0][0]
        self.__stop_token_idx = np.where(self.S == STOP_TOKEN)[0][0]
        self.__unknown_word_idx = np.where(self.S == unknown_words_default)[0][0]
        self.__known_words = set(emissions.reset_index().word.unique())
        self.__unknown_unit_vector = np.zeros(len(self.S))
        self.__unknown_unit_vector[self.__unknown_word_idx] = 1

    def predict(self, y_sentence):
        pi = np.zeros((len(y_sentence) + 1, len(self.S)))
        pi[0, self.__start_token_idx] = 1
        bp = np.zeros((len(y_sentence) + 1, len(self.S)), dtype=int)

        for k, word in enumerate(y_sentence):
            # dot product between row vectors pi[k-1] and transition[next_tag]
            tag_prob = (self.__q_prob * pi[k])
            best_previous_words = tag_prob.argmax(axis=1)
            tag_prob = tag_prob.max(axis=1)
            # dot product between two row vectors
            if word in self.__known_words:
                tag_prob = tag_prob * self.__e_prob.loc[word].values[:, 0]
            else:
                # e(x|NN) = 1, e(x|every_other) = 0
                tag_prob = tag_prob * self.__unknown_unit_vector
            pi[k + 1] = tag_prob  # the next_tag probability row
            bp[k + 1] = best_previous_words

        tags_sequence = self.__extract_path(pi, bp)
        return pd.DataFrame(zip(y_sentence, tags_sequence))

    def __extract_path(self, pi, bp):
        path = []
        trans = self.__q_prob[self.__stop_token_idx]
        n = pi.shape[0]  # for index use
        # last tag that maximizes the  transition  to stop
        path.append((pi[n-1] * trans).argmax().astype(int))

        # bp[0] is not relevant for the path (start token)
        for k in range(n-1, 1, -1):
            path.append(bp[k, path[-1]])  # the previous tag will be the one that
            # maximize the probability to move to the current tag.
        path.reverse()
        path = np.array(path, dtype=int)
        return self.S[path]
