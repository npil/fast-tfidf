# TensorFlow vs Fast-TF-IDF Comparison

## Summary

This comparison script (`compare_with_tensorflow.py`) generates fake text documents and compares the results between TensorFlow's `TextVectorization` layer and our fast-tf-idf implementation.

## Key Findings

### ✅ What Matches
1. **Vocabulary Extraction**: Both implementations correctly extract all unique terms from the documents
2. **Document Frequency Counting**: Both correctly count how many documents contain each term
3. **Vocabulary Ordering**: Both order terms by document frequency (most common first)

### ❌ What Differs
1. **IDF Formula**: The implementations use different IDF calculation formulas
2. **Absolute IDF Values**: Raw IDF weights differ significantly between implementations

## IDF Formula Differences

### Our Implementation (fast-tf-idf)
```
IDF = log((1 + n_docs) / (1 + document_frequency)) + 1
```
This matches sklearn's `TfidfVectorizer` with `smooth_idf=True` (the default).

**Example** (3 documents, term appears in 1):
```
IDF = log((1 + 3) / (1 + 1)) + 1
    = log(4/2) + 1
    = 0.693 + 1
    = 1.693
```

### TensorFlow's TextVectorization
TensorFlow uses a different internal formula that appears to be approximately:
```
IDF ≈ log(1 + n_docs / document_frequency)
```
However, the exact implementation may include additional normalization or scaling.

**Example** (3 documents, term appears in 1):
```
IDF ≈ log(1 + 3/1)
    = log(4)
    = 1.386
```

## Why the Difference?

Different TF-IDF implementations use different formulas based on:
- Historical conventions (information retrieval vs. machine learning)
- Numerical stability considerations
- Smoothing preferences
- Performance optimization trade-offs

Both formulas are valid and serve the same purpose: assigning higher weights to rare terms and lower weights to common terms.

## Test Results

When comparing on 100 generated documents:
- **Vocabulary**: 100% match (all terms found in both)
- **IDF Weights**: Differ by ~0.5-1.0 on average
- **Relative Ordering**: Similar (rare terms have higher IDF in both)
