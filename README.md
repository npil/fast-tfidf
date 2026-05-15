# Fast TF-IDF

A blazingly fast TF-IDF implementation for Python that uses multiprocessing to parallelise document processing.

**Key Features:**
- orders of magnitude faster than TensorFlow's TextVectorization (see benchmarks below)
- tensorflow-compatible vocabulary and idf weights can be used with TensorFlow's TextVectorization, it just computes them much quicker
- it's much more memory efficient too, consuming less than half the memory required by TextVectorization for the same dataset in my tests

## Installation

```bash
# Using pip
pip install fast-tfidf

# Or install from source
git clone https://github.com/npil/fast-tfidf.git
cd fast-tfidf
pip install -e .
```

## Usage

This package provides vocabulary extraction and IDF weight calculation that can be used standalone or integrated with other libraries.

### Basic Example

```python
from fast_tfidf import get_vocabulary_and_idf_weights

# Your text documents
documents = [
    "machine learning is amazing",
    "deep learning and machine learning",
    "artificial intelligence and deep learning"
]

# Extract vocabulary and IDF weights
# Get top 1000 terms (unigrams only)
vocabulary, idf_weights = get_vocabulary_and_idf_weights(
    documents,
    n_features=1000,
    remove_stopwords=False,
    max_ngrams=1
)

print(f"Vocabulary size: {len(vocabulary)}")
print(f"Top terms: {vocabulary[:5]}")
print(f"IDF weights: {idf_weights[:5]}")
```

### With Bigrams

```python
# Include bigrams for richer representation
vocabulary, idf_weights = get_vocabulary_and_idf_weights(
    documents,
    n_features=2000,
    max_ngrams=2  # Include bigrams
)
```

### With Stopword Removal

```python
# Remove common English stopwords
vocabulary, idf_weights = get_vocabulary_and_idf_weights(
    documents,
    n_features=1000,
    remove_stopwords=True  # Filter out stopwords
)
```

### Integration with TensorFlow

```python
from tensorflow.keras.layers import TextVectorization
from fast_tfidf import get_vocabulary_and_idf_weights

vocabulary, idf_weights = get_vocabulary_and_idf_weights(documents, n_features=2000)

vectoriser = TextVectorization(
    vocabulary=vocabulary,
    idf_weights=idf_weights,
    output_mode='tf_idf'
)

# No adapt() needed - use pre-computed weights! 🚀
# Your model can now use 'vectoriser' for TF-IDF transformation
```

**Note:** Although fast-tf-idf uses a different IDF formula than TensorFlow's default `log((1 + n_docs) / (1 + df)) + 1` vs TensorFlow's internal formula), this only affects absolute values while preserving relative term importance.

## Performance

Benchmark results comparing against TensorFlow's TextVectorization:

| Documents | TensorFlow | fast-tfidf | **Speedup** |
|-----------|------------|-------------|-------------|
| 100       | 204 ms     | 29 ms       | **7.1x** ⚡ |
| 1,000     | 1.06 s     | 28 ms       | **38.3x** ⚡⚡ |
| 5,000     | 3.00 s     | 43 ms       | **70.4x** ⚡⚡⚡ |
| 10,000    | 7.51 s     | 77 ms       | **97.0x** ⚡⚡⚡⚡ |

*See `benchmarks/` directory for detailed results and comparison scripts.*


## Development
Dependencies are managed using [poetry](https://python-poetry.org).

There is a [vscode dev container](https://code.visualstudio.com/docs/devcontainers/containers) that can be used for development, or otherwise you can run `make install` in the environment of your choice to install the package and all dependencies (requires poetry and [make](https://www.gnu.org/software/make/)).

We use the following tools to ensure code quality:
1. [ruff](https://github.com/astral-sh/ruff)
4. [mypy](https://mypy.readthedocs.io/en/stable/)
5. [pytest](https://docs.pytest.org/en/stable/)

You can run all formatting / linting / type checks using `make lint` (you can use `make fix` to automatically fix some formatting errors).

You can run unittests using `make test`.

You can run linting and unittests using `make lint-test`.