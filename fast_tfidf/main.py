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
    max_ngrams: int = 1,
) -> collections.Counter:
    word_count: collections.Counter = collections.Counter()

    for document in text:
        tokens = split_sentence(str(document))
        if remove_stopwords:
            tokens = [t for t in tokens if t not in STOP_WORDS]

        word_count.update(set(tokens))

        for n in range(2, max_ngrams + 1):
            ngrams = zip(*[tokens[i:] for i in range(n)], strict=False)
            word_count.update(set(" ".join(ngram) for ngram in ngrams))

    return word_count


def _count_tokens_worker(
    args: tuple[list[str], bool, int],
) -> collections.Counter[str]:
    """Worker function for multiprocessing that can be pickled."""
    return count_tokens(*args)


def count_tokens_sharded(
    text: typing.Sequence[str], remove_stopwords: bool = False, max_ngrams: int = 1
) -> list[collections.Counter[str]]:
    cpu_count = multiprocessing.cpu_count()

    text_splits = numpy.array_split(text, cpu_count)
    args_list: list[tuple[list[str], bool, int]] = [
        (split.tolist() if hasattr(split, "tolist") else list(split), remove_stopwords, max_ngrams)
        for split in text_splits
    ]

    with multiprocessing.Pool(cpu_count) as p:
        return p.map(_count_tokens_worker, args_list)


def get_vocabulary_and_idf_weights(
    text: typing.Sequence[str],
    n_features: int | None = None,
    remove_stopwords: bool = False,
    max_ngrams: int = 1,
) -> tuple[list[str], list[float]]:
    counters = count_tokens_sharded(text, remove_stopwords=remove_stopwords, max_ngrams=max_ngrams)

    counter_overall: collections.Counter[str] = collections.Counter()
    _ = [counter_overall.update(c) for c in counters]

    n_text = len(text)

    vocabulary = [x[0] for x in counter_overall.most_common(n_features)]
    idf_weights = [math.log((1 + n_text) / (1 + counter_overall[term])) + 1 for term in vocabulary]

    return vocabulary, idf_weights


def split_sentence(sentence: str) -> list[str]:
    return [x.lower() for x in re.split(r"\W+", sentence.strip()) if x]
