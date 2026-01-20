"""
Performance benchmark comparing TensorFlow's TextVectorization with fast-tfidf.
Measures execution time with various document corpus sizes.
"""

import random
import time
from statistics import mean, stdev
from typing import Any

import numpy as np
from tensorflow.keras.layers import TextVectorization

from fast_tfidf.main import get_vocabulary_and_idf_weights


def generate_realistic_documents(
    n_docs: int = 1000, words_per_doc_range: tuple[int, int] = (10, 50), vocabulary_size: int = 1000
) -> list[str]:
    """
    Generate realistic-looking text documents for benchmarking.

    Args:
        n_docs: Number of documents to generate
        words_per_doc_range: Tuple of (min, max) words per document
        vocabulary_size: Size of the vocabulary pool to draw from

    Returns:
        List of generated documents (strings)
    """
    # Create a pool of fake words with Zipfian distribution (realistic word frequencies)
    word_pool = []

    # Very common words (top 10%)
    very_common = [f"word{i}" for i in range(vocabulary_size // 10)]
    word_pool.extend(very_common * 20)

    # Common words (next 20%)
    common = [f"term{i}" for i in range(vocabulary_size // 5)]
    word_pool.extend(common * 8)

    # Medium frequency words (next 30%)
    medium = [f"token{i}" for i in range(int(vocabulary_size * 0.3))]
    word_pool.extend(medium * 3)

    # Rare words (remaining 40%)
    rare = [f"rare{i}" for i in range(int(vocabulary_size * 0.4))]
    word_pool.extend(rare)

    documents = []
    for _ in range(n_docs):
        n_words = random.randint(*words_per_doc_range)
        doc_words = random.choices(word_pool, k=n_words)
        documents.append(" ".join(doc_words))

    return documents


def benchmark_tensorflow(
    documents: list[str], max_tokens: int | None = None, n_runs: int = 3
) -> dict[str, Any]:
    """Benchmark TensorFlow's TextVectorization."""
    times: list[float] = []

    for _ in range(n_runs):
        start_time = time.time()

        vectorizer = TextVectorization(
            max_tokens=max_tokens,
            output_mode="tf_idf",
            standardize="lower_and_strip_punctuation",
            split="whitespace",
            ngrams=None,
        )
        vectorizer.adapt(documents)

        vocab = [term for term in vectorizer.get_vocabulary() if term and term != "[UNK]"]

        # Extract IDF weights for all terms
        idf_weights = []
        for term in vocab:
            test_doc = [term]
            result = vectorizer(test_doc)
            term_idx = vectorizer.get_vocabulary().index(term)
            idf_weight = float(result[0, term_idx].numpy())
            idf_weights.append(idf_weight)

        elapsed = time.time() - start_time
        times.append(elapsed)

    return {
        "mean": mean(times),
        "stdev": stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
        "vocab_size": len(vocab),
        "all_times": times,
    }


def benchmark_fast_tfidf(
    documents: list[str], max_tokens: int | None = None, n_runs: int = 3
) -> dict[str, Any]:
    times: list[float] = []

    for _ in range(n_runs):
        start_time = time.time()

        vocab, idf_weights = get_vocabulary_and_idf_weights(
            documents, n_features=max_tokens, remove_stopwords=False, use_bigrams=False
        )

        elapsed = time.time() - start_time
        times.append(elapsed)

    return {
        "mean": mean(times),
        "stdev": stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
        "vocab_size": len(vocab),
        "all_times": times,
    }


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def run_benchmark(
    n_docs: int,
    words_per_doc_range: tuple[int, int],
    vocabulary_size: int,
    max_tokens: int | None = None,
    n_runs: int = 3,
) -> dict[str, Any]:
    """Run a single benchmark comparison."""
    print(f"\n{'=' * 70}")
    print(
        f"Benchmark: {n_docs:,} documents, {words_per_doc_range[0]}-{words_per_doc_range[1]} words/doc"
    )
    print(f"Vocabulary size: {vocabulary_size:,}, Max tokens: {max_tokens or 'unlimited'}")
    print(f"{'=' * 70}")

    # Generate documents
    print("Generating documents...")
    random.seed(42)
    np.random.seed(42)
    documents = generate_realistic_documents(n_docs, words_per_doc_range, vocabulary_size)
    print(f"Generated {len(documents):,} documents")
    print(f"Sample: {documents[0][:100]}...")

    # Benchmark TensorFlow
    print(f"\n{'TensorFlow TextVectorization':-^70}")
    tf_results = benchmark_tensorflow(documents, max_tokens, n_runs)
    print(f"Mean time:    {format_time(tf_results['mean'])}")
    print(f"Std dev:      {format_time(tf_results['stdev'])}")
    print(f"Min time:     {format_time(tf_results['min'])}")
    print(f"Max time:     {format_time(tf_results['max'])}")
    print(f"Vocab size:   {tf_results['vocab_size']:,}")
    print(f"All runs:     {', '.join([format_time(t) for t in tf_results['all_times']])}")

    # Benchmark fast-tf-idf
    print(f"\n{'fast-tf-idf Implementation':-^70}")
    our_results = benchmark_fast_tfidf(documents, max_tokens, n_runs)
    print(f"Mean time:    {format_time(our_results['mean'])}")
    print(f"Std dev:      {format_time(our_results['stdev'])}")
    print(f"Min time:     {format_time(our_results['min'])}")
    print(f"Max time:     {format_time(our_results['max'])}")
    print(f"Vocab size:   {our_results['vocab_size']:,}")
    print(f"All runs:     {', '.join([format_time(t) for t in our_results['all_times']])}")

    # Comparison
    print(f"\n{'Comparison':-^70}")
    speedup = tf_results["mean"] / our_results["mean"]
    if speedup > 1:
        print(f"✓ fast-tf-idf is {speedup:.2f}x FASTER than TensorFlow")
    else:
        print(f"✗ fast-tf-idf is {1 / speedup:.2f}x SLOWER than TensorFlow")

    print(f"\nAbsolute difference: {format_time(abs(tf_results['mean'] - our_results['mean']))}")

    return {"n_docs": n_docs, "tf_results": tf_results, "our_results": our_results}


def main() -> None:
    """Main benchmark function."""
    print("=" * 70)
    print("PERFORMANCE BENCHMARK: TensorFlow vs fast-tf-idf")
    print("=" * 70)
    print("\nThis benchmark compares execution time for vocabulary extraction")
    print("and IDF weight calculation on various corpus sizes.")
    print("=" * 70)

    # Run benchmarks with increasing document counts
    benchmark_configs = [
        # (n_docs, words_per_doc_range, vocabulary_size, max_tokens, n_runs)
        (100, (10, 30), 100, None, 5),
        (1000, (10, 50), 500, None, 5),
        (5000, (10, 50), 1000, None, 3),
        (10000, (10, 50), 2000, None, 3),
    ]

    results = []

    for n_docs, word_range, vocab_size, max_tokens, n_runs in benchmark_configs:
        result = run_benchmark(n_docs, word_range, vocab_size, max_tokens, n_runs)
        results.append(result)
        print()  # Blank line between benchmarks

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Documents':<12} {'TensorFlow':<20} {'fast-tf-idf':<20} {'Speedup':<15}")
    print("-" * 70)

    for result in results:
        n_docs_val = result["n_docs"]
        tf_results_val = result["tf_results"]
        our_results_val = result["our_results"]
        tf_time = format_time(tf_results_val["mean"])
        our_time = format_time(our_results_val["mean"])
        speedup = tf_results_val["mean"] / our_results_val["mean"]
        speedup_str = f"{speedup:.2f}x" if speedup > 1 else f"-{1 / speedup:.2f}x"

        print(f"{n_docs_val:<12,} {tf_time:<20} {our_time:<20} {speedup_str:<15}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if results:
        speedups = []
        for r in results:
            tf_r = r["tf_results"]
            our_r = r["our_results"]
            speedups.append(tf_r["mean"] / our_r["mean"])
        avg_speedup = mean(speedups)

        if avg_speedup > 1:
            print(f"On average, fast-tf-idf is {avg_speedup:.2f}x FASTER than TensorFlow")
            print("for vocabulary extraction and IDF weight calculation.")
        else:
            print(f"On average, fast-tf-idf is {1 / avg_speedup:.2f}x SLOWER than TensorFlow")
            print("for vocabulary extraction and IDF weight calculation.")

        print("\nNote: These benchmarks measure only the vocabulary and IDF extraction.")
        print("Full vectorization (converting documents to vectors) was not measured.")

    print("=" * 70)


if __name__ == "__main__":
    main()
