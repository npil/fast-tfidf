import math

from fast_tfidf.main import count_tokens, get_vocabulary_and_idf_weights, split_sentence


class TestSplitSentence:
    """Tests for the split_sentence function."""

    def test_basic_sentence(self):
        result = split_sentence("Hello World")
        assert result == ["hello", "world"]

    def test_punctuation_removal(self):
        result = split_sentence("Hello, World! How are you?")
        assert result == ["hello", "world", "how", "are", "you"]

    def test_empty_string(self):
        result = split_sentence("")
        assert result == []

    def test_numbers_and_special_chars(self):
        result = split_sentence("Test123 @#$ Example456")
        assert result == ["test123", "example456"]

    def test_whitespace_handling(self):
        result = split_sentence("  multiple   spaces  ")
        assert result == ["multiple", "spaces"]


class TestCountTokens:
    """Tests for the count_tokens function."""

    def test_simple_documents(self):
        documents = ["hello world", "world test"]
        result = count_tokens(documents)

        # Check unigrams
        assert result["hello"] == 1  # appears in 1 document
        assert result["world"] == 2  # appears in 2 documents
        assert result["test"] == 1  # appears in 1 document

    def test_bigrams(self):
        documents = ["hello world", "hello world test"]
        result = count_tokens(documents, use_bigrams=True)

        # Check bigrams
        assert result["hello world"] == 2  # bigram appears in 2 documents
        assert result["world test"] == 1  # bigram appears in 1 document    def test_remove_stopwords_false(self):
        documents = ["the cat sat on the mat", "the dog ran"]
        result = count_tokens(documents, remove_stopwords=False)

        # Stopwords should be included
        assert "the" in result
        assert "on" in result
        assert result["the"] == 2  # appears in both documents

    def test_remove_stopwords_true(self):
        documents = ["the cat sat on the mat", "the dog ran"]
        result = count_tokens(documents, remove_stopwords=True)

        # Stopwords should be excluded
        assert "the" not in result
        assert "on" not in result
        # Content words should be included
        assert "cat" in result
        assert "dog" in result
        assert "mat" in result

    def test_document_frequency_not_term_frequency(self):
        # Word "test" appears multiple times in one document
        documents = ["test test test", "other word"]
        result = count_tokens(documents)

        # Should count document frequency (1), not term frequency (3)
        assert result["test"] == 1

    def test_empty_documents(self):
        documents = []
        result = count_tokens(documents)
        assert len(result) == 0

    def test_single_document(self):
        documents = ["hello world"]
        result = count_tokens(documents, use_bigrams=True)
        assert result["hello"] == 1
        assert result["world"] == 1
        assert result["hello world"] == 1


class TestGetVocabularyAndIdfWeights:
    """Tests for the get_vocabulary_and_idf_weights function."""

    def test_basic_vocabulary_and_idf(self):
        documents = ["hello world", "world test", "hello test"]
        vocabulary, idf_weights = get_vocabulary_and_idf_weights(documents)

        assert "hello" in vocabulary
        assert "world" in vocabulary
        assert "test" in vocabulary

        n_docs = 3
        for i, term in enumerate(vocabulary):
            if term in ["hello", "world", "test"]:
                # Each appears in 2 documents
                expected_idf = math.log((1 + n_docs) / (1 + 2)) + 1
                assert abs(idf_weights[i] - expected_idf) < 1e-10

    def test_n_features_limit(self):
        documents = ["apple banana cherry date", "apple banana", "apple"]
        vocabulary, idf_weights = get_vocabulary_and_idf_weights(documents, n_features=2)

        assert len(vocabulary) == 2
        assert len(idf_weights) == 2
        # "apple" should be most common (appears in 3 docs)
        assert vocabulary[0] == "apple"

    def test_idf_formula_matches_tensorflow(self):
        # Test that IDF formula matches TensorFlow's: log((1 + n_docs) / (1 + df)) + 1
        documents = ["word1 word2", "word1 word3", "word1 word4"]
        vocabulary, idf_weights = get_vocabulary_and_idf_weights(documents)

        n_docs = 3
        word1_idx = vocabulary.index("word1")
        # word1 appears in 3 documents
        expected_idf_word1 = math.log((1 + n_docs) / (1 + 3)) + 1
        assert abs(idf_weights[word1_idx] - expected_idf_word1) < 1e-10

        word2_idx = vocabulary.index("word2")
        # word2 appears in 1 document
        expected_idf_word2 = math.log((1 + n_docs) / (1 + 1)) + 1
        assert abs(idf_weights[word2_idx] - expected_idf_word2) < 1e-10

    def test_remove_stopwords_parameter(self):
        documents = ["the cat is here", "the dog is there"]

        # Without stopword removal
        vocab_with_stopwords, _ = get_vocabulary_and_idf_weights(documents, remove_stopwords=False)
        assert "the" in vocab_with_stopwords
        assert "is" in vocab_with_stopwords

        # With stopword removal
        vocab_without_stopwords, _ = get_vocabulary_and_idf_weights(documents, remove_stopwords=True)
        assert "the" not in vocab_without_stopwords
        assert "is" not in vocab_without_stopwords
        assert "cat" in vocab_without_stopwords
        assert "dog" in vocab_without_stopwords

    def test_bigrams_in_vocabulary(self):
        documents = ["hello world", "hello world test"]
        vocabulary, idf_weights = get_vocabulary_and_idf_weights(documents, use_bigrams=True)

        # Bigrams should be included
        assert "hello world" in vocabulary
        assert "world test" in vocabulary

    def test_vocabulary_sorted_by_frequency(self):
        documents = [
            "common common common",
            "common common",
            "common rare",
        ]
        vocabulary, _ = get_vocabulary_and_idf_weights(documents, n_features=2)

        # Most frequent terms should come first
        # "common" appears in 3 docs, "rare" in 1 doc
        assert vocabulary[0] == "common"

    def test_return_types(self):
        documents = ["test document"]
        vocabulary, idf_weights = get_vocabulary_and_idf_weights(documents)

        assert isinstance(vocabulary, list)
        assert isinstance(idf_weights, list)
        assert len(vocabulary) == len(idf_weights)
        assert all(isinstance(term, str) for term in vocabulary)
        assert all(isinstance(weight, float) for weight in idf_weights)

    def test_empty_documents_list(self):
        documents = []
        vocabulary, idf_weights = get_vocabulary_and_idf_weights(documents)

        assert vocabulary == []
        assert idf_weights == []

    def test_single_document_idf(self):
        documents = ["hello world test"]
        vocabulary, idf_weights = get_vocabulary_and_idf_weights(documents)

        # With only 1 document, all terms appear in 1 document
        # IDF should be: log((1 + 1) / (1 + 1)) + 1 = log(1) + 1 = 0 + 1 = 1
        expected_idf = math.log((1 + 1) / (1 + 1)) + 1
        for weight in idf_weights:
            assert abs(weight - expected_idf) < 1e-10


class TestIntegration:
    """Integration tests comparing with expected TensorFlow-like behavior."""

    def test_realistic_corpus(self):
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "Never jump over the lazy dog quickly",
            "The fox is quick and smart",
        ]

        vocabulary, idf_weights = get_vocabulary_and_idf_weights(documents, n_features=10, remove_stopwords=True)

        # Check that we got vocabulary and weights
        assert len(vocabulary) > 0
        assert len(vocabulary) == len(idf_weights)
        assert len(vocabulary) <= 10

        # All IDF weights should be positive
        assert all(weight > 0 for weight in idf_weights)

        # Check that content words are present
        assert any(word in vocabulary for word in ["quick", "fox", "dog", "lazy"])

    def test_consistency_across_calls(self):
        documents = ["hello world", "world test", "hello test"]

        vocab1, idf1 = get_vocabulary_and_idf_weights(documents)
        vocab2, idf2 = get_vocabulary_and_idf_weights(documents)

        # Results should be consistent
        assert vocab1 == vocab2
        assert idf1 == idf2

    def test_idf_decreases_with_frequency(self):
        # More frequent terms should have lower IDF
        documents = [
            "common common common",
            "common common rare",
            "common rare very_rare",
        ]

        vocabulary, idf_weights = get_vocabulary_and_idf_weights(documents)

        common_idx = vocabulary.index("common")
        rare_idx = vocabulary.index("rare")
        very_rare_idx = vocabulary.index("very_rare")

        # common appears in 3 docs, rare in 2, very_rare in 1
        # IDF should be: common < rare < very_rare
        assert idf_weights[common_idx] < idf_weights[rare_idx]
        assert idf_weights[rare_idx] < idf_weights[very_rare_idx]
