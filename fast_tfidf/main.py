import collections
import math
import multiprocessing
import re
import typing

import nltk
import numpy

nltk.download("stopwords")
STOP_WORDS = set(nltk.corpus.stopwords.words("english"))


def count_tokens(
    text: typing.Sequence[str],
    remove_stopwords: bool = False,
    use_bigrams: bool = False,
) -> collections.Counter:
    word_count = collections.Counter()

    for document in text:
        tokens = split_sentence(str(document))
        if remove_stopwords:
            tokens = [t for t in tokens if t not in STOP_WORDS]

        word_count.update(set(tokens))

        if use_bigrams:
            bigrams = zip(tokens, tokens[1:])
            word_count.update(set([" ".join(bigram) for bigram in bigrams]))

    return word_count


def _count_tokens_worker(args):
    """Worker function for multiprocessing that can be pickled."""
    text_chunk, kwargs = args
    return count_tokens(text_chunk, **kwargs)


def count_tokens_sharded(text: typing.Sequence[str], **kwargs: bool) -> list[collections.Counter]:
    cpu_count = multiprocessing.cpu_count()

    text_splits = numpy.array_split(text, cpu_count)
    args_list = [(split.tolist() if hasattr(split, "tolist") else split, kwargs) for split in text_splits]

    with multiprocessing.Pool(cpu_count) as p:
        return p.map(_count_tokens_worker, args_list)


def get_vocabulary_and_idf_weights(
    text: typing.Sequence[str],
    n_features: int | None = None,
    remove_stopwords: bool = False,
    use_bigrams: bool = False,
) -> list[str]:
    counters = count_tokens_sharded(text, remove_stopwords=remove_stopwords, use_bigrams=use_bigrams)

    counter_overall = collections.Counter()
    for c in counters:
        counter_overall.update(c)

    n_text = len(text)

    vocabulary = [x[0] for x in counter_overall.most_common(n_features)]
    idf_weights = [math.log((1 + n_text) / (1 + counter_overall[term])) + 1 for term in vocabulary]

    return vocabulary, idf_weights


def split_sentence(sentence):
    return [x.lower() for x in re.split(r"\W+", sentence.strip()) if x]
