import numpy as np
# brown = nltk.download('brown')
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import warnings

warnings.filterwarnings("ignore")
from nltk.corpus import brown

ASTRIX, START_TOKEN, STOP_TOKEN = "*", "START", "STOP"
MONTHS_CAPS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
               'September', 'October', 'November', 'December']
MONTHS_LOWER = [m.lower() for m in MONTHS_CAPS]
PSEUDO_DICT = {'Dollars_Amount': ['\$', '\d-cent'],
               'Size_Related': ['\d+\Wacre', '\d+\Wsize', '\d+\Wdegree', '\d+\Wyard',
                                '\d+\Wounce', '\d+\Wfoot', '\d+\Wpiece',
                                '\d+\Wunit', '\d+\Winch', '\d+\Wpound', '\d+\Wpiece'],
               'Monetary_Amount': ['[0-9]+\.[0-9]+', '[0-9]*(,[0-9\,]*)', '\d+%', '\d/'],
               'Time_Related': ['hour', 'minute', '\d:\d\d', '\d\'s'],
               'Age_Related': ['\Wage', '\Wold', 'year-old', '-year'],
               'Calender': ['day', '\dth', 'month', '[0-9]+[-]*[st,nd]'] + MONTHS_CAPS +
                           MONTHS_LOWER,
               'Two&Four_Digits': ['\w*[1-9]{1}[0-9]{3}\w*', '\w*[1-9]{1}[0-9]{1}\w*'],
               'Groups': ['[0-9]-[a-z]'],
               'Other_Number': ['\d'],
               'Synonym': ['[A-Z]\.'],
               'All_Cap': ['^[A-Z]+$'],
               'Init_Cap': ['^[A-Z][a-z]+'],
               'Low_Case': ['^[a-z]+$'],
               'Other': ['']}


def load_sentences_data():
    sentences = list(brown.tagged_sents(categories='news'))
    n = len(sentences)
    test_size = round(n * 0.1)
    train_size = n - test_size
    train_sent, test_sent = sentences[:train_size], sentences[-test_size:]
    print("Loaded sentences:")
    print(f"train size: {len(train_sent)} sentences")
    print(f"test size: {len(test_sent)} sentences")
    print()
    return train_sent, test_sent


def consider_tag_prefix(tag_series):
    result = tag_series.apply(lambda x: x.split("+")[0].split("-")[0])
    return result


def get_word_tags(sentences_nltk):
    return [(word, tag) for sent in sentences_nltk for (word, tag) in sent]


def adapt_data_by_words(train_sentences, test_sentences):
    train_words = get_word_tags(train_sentences)
    train_words = pd.DataFrame(train_words, columns=['word', 'tag'])
    train_words.tag = consider_tag_prefix(train_words.tag)

    test_words = get_word_tags(test_sentences)
    test_words = pd.DataFrame(test_words, columns=['word', 'tag'])
    test_words.tag = consider_tag_prefix(test_words.tag)
    print("Words in sentences: ")
    print(
        f"train size: {len(train_words)} words; {train_words.word.nunique()} wordforms, "
        f"{train_words.tag.nunique()} distinct tags")
    print(
        f"test size: {len(test_words)} words; {test_words.word.nunique()} wordforms, "
        f"{test_words.tag.nunique()} distinct tags")
    print()
    print("Train-Test relations:")
    print(
        f"{np.isin(test_words.word.unique(), train_words.word.unique()).sum()} "
        f"words from test appears in train")
    print(
        f"{np.isin(test_words.tag.unique(), train_words.tag.unique()).sum()} "
        f"tags from test appears in train")
    print()
    return train_words, test_words


def adapt_sentences_tokens(sentences):
    """Mainly add Start and End tokens, and creates the next_tag column (by sentence
    perspective)."""
    rows = []
    for i, sent in enumerate(sentences):
        rows.append({'word': ASTRIX, 'tag': START_TOKEN, 'sent_id': i})
        for word, tag in sent:
            rows.append({'word': word, 'tag': tag, 'sent_id': i})
        rows.append({'word': ASTRIX, 'tag': STOP_TOKEN, 'sent_id': i})
    df = pd.DataFrame(rows)
    df.tag = consider_tag_prefix(df.tag)
    return df


def adapt_test(sentences):
    rows = []
    for i, sent in enumerate(sentences):
        for word, tag in sent:
            rows.append({'word': word, 'tag': tag, 'sent_id': i})
    df = pd.DataFrame(rows)
    df.tag = consider_tag_prefix(df.tag)
    return df


def add_next_tag_col(df):
    next_tag = df.tag[1:]
    next_tag.index = next_tag.index - 1
    df['next_tag'] = next_tag
    return df


def adapt_pseudo_words(df):
    df = df[['word']]
    for category in PSEUDO_DICT.keys():
        mask = df.word.str.contains("|".join(PSEUDO_DICT[category]))
        mask = mask.replace({True: category, False: np.nan})
        df[category] = mask
    df['other'] = ['other'] * df.shape[0]
    df.insert(1, 'pseudo_words', np.nan)
    return df.fillna(method='bfill', axis=1)['pseudo_words']
