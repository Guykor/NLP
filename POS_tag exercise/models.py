from sentences_parser import adapt_sentences_tokens, adapt_pseudo_words, add_next_tag_col, \
    adapt_test
from viterby_algo import *
import time
UNKNOWN_WORDS_TAG = "NN"


def compute_error_rate(y, pred_y):
    accuracy = (y == pred_y).mean()
    return 1 - accuracy


def print_error_rates(evaluate, train_words):
    """
    :param evaluate: df with test_word, tag_pred, tag_label columns
    :param test_words: raw words from test set, contains column word
    """
    known = evaluate[evaluate.word.isin(train_words.word)]
    unknown = evaluate[~evaluate.word.isin(train_words.word)]

    print("* Error rate for known words: ",
          compute_error_rate(known.tag_label, known.tag_pred))

    print("* Error rate for unknown words: ",
          compute_error_rate(unknown.tag_label, unknown.tag_pred))

    print("* Total Error rate: ", compute_error_rate(evaluate.tag_label, evaluate.tag_pred))
    print()


class MLE:
    def __init__(self):
        self.mle_tbl = None

    def fit(self, train_words):
        bag = train_words.copy()
        bag['word_tag_count'] = bag.reset_index() \
            .groupby(['word', 'tag'])['index'].transform('count')
        bag['word_count'] = bag.groupby('word')['tag'].transform('count')
        bag['MLE'] = bag.word_tag_count / bag.word_count
        bag = bag.drop_duplicates(['word', 'tag'])
        self.mle_tbl = bag.groupby(['word'])[['tag', 'MLE']].max()['tag'].reset_index()
        print(
            f'MLE model trained on {bag.word.nunique()} unique words, and {bag.tag.nunique()} '
            f'unique tags\n')

    def predict(self, test_words):
        """assuming test words is a container of string"""
        return pd.merge(test_words, self.mle_tbl, on='word', how='left').fillna(UNKNOWN_WORDS_TAG)

    def evaluate(self, test_words):
        """Assuming that the test_words is a data frame with columns ['word', 'tag']"""
        result = pd.merge(test_words, self.mle_tbl, on='word', how='left',
                          suffixes=['_label', '_pred']).fillna(UNKNOWN_WORDS_TAG)
        print("MLE model evaluation:\n")
        print_error_rates(result, self.mle_tbl)


class HMM:
    def __init__(self, laplace_smooth_factor=0, create_pseudo_words_threshold=0):
        self.__transitions = None
        self.__emissions = None
        self.__predictor = None
        self.__known_words = None
        self.__laplace_smoother = laplace_smooth_factor
        self.__pseudo_threshold = create_pseudo_words_threshold

    def __create_emission_tbl(self, tag_frequency, word_tag_freq):
        """calculated with laplace smoothing, add-0 as default."""
        emissions = pd.merge(tag_frequency, word_tag_freq, on='tag', how='right')
        V = word_tag_freq.word.nunique()
        emissions['emission_prob'] = (emissions.word_tag_frequency + self.__laplace_smoother) / \
                                     (emissions.tag_frequency + (self.__laplace_smoother * V))
        emissions = emissions.drop(columns=['tag_frequency', 'word_tag_frequency'])
        emissions = emissions.drop_duplicates().set_index(['word', 'tag'])
        return emissions

    def fit(self, train_sentences):
        hmm_mle = adapt_sentences_tokens(train_sentences)
        hmm_mle = add_next_tag_col(hmm_mle)
        if self.__pseudo_threshold:
            freq_words = hmm_mle.groupby('word')[['tag']].transform('count')
            freq_words.columns = ['freq']
            adapted = adapt_pseudo_words(
                hmm_mle[freq_words.freq <= self.__pseudo_threshold][['word']])
            hmm_mle.loc[freq_words.freq <= self.__pseudo_threshold, 'word'] = adapted

        joint_tag_frequency = hmm_mle.groupby(['tag', 'next_tag'])[['word']].count() \
            .rename(columns={'word': 'joint_frequency'}).reset_index()
        tag_frequency = hmm_mle.groupby('tag')[['word']].count().reset_index() \
            .rename(columns={'word': 'tag_frequency'})

        transitions = pd.merge(tag_frequency, joint_tag_frequency, on='tag', how='right')
        transitions['transition_prob'] = transitions.joint_frequency / transitions.tag_frequency
        transitions = transitions.drop(columns=['tag_frequency', 'joint_frequency'])
        transitions = transitions.drop_duplicates().set_index(['next_tag', 'tag'])

        # zero garbage transitions that were created in process -
        # * -> START (when start is in next_tag), and STOP->* (when STOP is in tag column)
        # its the same case
        transitions.loc[START_TOKEN] = 0

        self.__transitions = transitions

        word_tag_freq = hmm_mle.reset_index().groupby(['word', 'tag'])[['index']].count() \
            .rename(columns={'index': 'word_tag_frequency'}).reset_index()

        self.__emissions = self.__create_emission_tbl(tag_frequency, word_tag_freq)
        self.__emissions.to_csv("emiss.csv")
        self.__predictor = Viterby(self.__transitions, self.__emissions,
                                   unknown_words_default=UNKNOWN_WORDS_TAG)

    def evaluate(self, test_set):
        test_set = adapt_test(test_set)
        test_sentences, test_labels = test_set[['word', 'sent_id']], test_set[['tag', 'sent_id']]
        predictions = []
        start = time.time()
        print("HMM predicting....")
        test_sentences['original_word'] = test_sentences.word
        if self.__pseudo_threshold:
            is_unknown_word = ~test_sentences.word.isin(self.__emissions.reset_index().word.unique())
            adapted = adapt_pseudo_words(test_sentences.loc[is_unknown_word])
            test_sentences.loc[is_unknown_word, 'word'] = adapted

        for i, sentence_words in test_sentences.groupby('sent_id'):
            pred = self.__predictor.predict(sentence_words.word)
            # save as words for evaluation by words and their tags
            pred['label'] = test_labels.loc[test_labels.sent_id == i, 'tag'].values
            predictions.append(pred)
        print("Evaluating HMM on test set took ", (time.time() - start) / 60, " minutes.\n")
        evaluate = pd.concat(predictions)
        evaluate.columns = ['word', 'tag_pred', 'tag_label']
        evaluate.word = test_sentences.original_word
        print_error_rates(evaluate, self.__emissions.reset_index())
        return evaluate

    def confusion_matrix(self, evaluation_df):
        confusion_matrix = pd.crosstab(evaluation_df.tag_label,
                                       evaluation_df.tag_pred,
                                       rownames=['Actual'],
                                       colnames=['Predicted'])
        confusion_matrix = confusion_matrix.stack()
        confusion_matrix = confusion_matrix[confusion_matrix != 0].sort_values(
            ascending=False).reset_index()
        confusion_matrix.columns = ['Actual', 'Predicted', 'Occurences']
        print(confusion_matrix)


