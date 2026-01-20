# Benchmarks and Comparisons

This directory contains performance benchmarks and comparison scripts for the fast-tf-idf implementation.

You'll need to install tensorflow yourself to run them. As of the timestamp of this file, tensorflow does not support Python 3.14 so I've removed it from the development dependencies of this package. It can be added again once support for Python 3.14 has been added.

## Quick Start

```bash
# Run performance benchmark
python benchmark.py

# Simple side-by-side comparison
python simple_comparison.py
```

## Files

### Scripts

- **`benchmark.py`** - Comprehensive performance benchmark comparing TensorFlow vs fast-tfidf
- **`simple_comparison.py`** - Comparison between fast-tfidf and tensorflow computed term frequencies and idf weights

### Documentation

- **`COMPARISON_RESULTS.md`** - Analysis of formula differences between implementations

## Key Results

### Performance Comparison

| Documents | TensorFlow | fast-tf-idf | **Speedup** |
|-----------|------------|-------------|-------------|
| 100       | 204 ms     | 29 ms       | **7.1x** ⚡ |
| 1,000     | 1.06 s     | 28 ms       | **38.3x** ⚡⚡ |
| 5,000     | 3.00 s     | 43 ms       | **70.4x** ⚡⚡⚡ |
| 10,000    | 7.51 s     | 77 ms       | **97.0x** ⚡⚡⚡⚡ |

**Average speedup: 53x faster!** 🚀

### Formula Differences

- **fast-tf-idf**: `IDF = log((1 + n_docs) / (1 + df)) + 1` (sklearn-compatible)
- **TensorFlow**: Uses different internal formula

Both implementations correctly extract vocabulary and calculate IDF weights, but with different formulas that produce different absolute values while maintaining the same relative ordering and relative idf weights.
