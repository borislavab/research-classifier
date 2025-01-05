from .dataset import extract_labels, LABELS
import numpy as np


def test_extract_labels_single_category():
    # Test with a single category that exists in LABELS
    categories = "cs.AI"
    result = extract_labels(categories)

    # Should be all zeros except for cs.AI position
    expected = [0] * len(LABELS)
    expected[LABELS.index("cs.AI")] = 1

    np.testing.assert_array_equal(result, expected)
    assert sum(result) == 1  # Only one category should be marked


def test_extract_labels_multiple_categories():
    # Test with multiple valid categories
    categories = "cs.AI cs.LG"
    result = extract_labels(categories)

    expected = [0] * len(LABELS)
    expected[LABELS.index("cs.AI")] = 1
    expected[LABELS.index("cs.LG")] = 1

    np.testing.assert_array_equal(result, expected)
    assert sum(result) == 2  # Two categories should be marked


def test_extract_labels_empty_string():
    # Test with empty string
    categories = ""
    result = extract_labels(categories)

    expected = [0] * len(LABELS)

    np.testing.assert_array_equal(result, expected)
    assert sum(result) == 0  # No categories should be marked


def test_extract_labels_batch():
    # Test with a batch of category strings
    categories = ["cs.AI", "cs.AI cs.LG", "", "cs.LG"]
    result = extract_labels(categories)

    # Expected results for each input
    expected = np.zeros((4, len(LABELS)), dtype=np.float32)

    # First sample: "cs.AI"
    expected[0, LABELS.index("cs.AI")] = 1.0

    # Second sample: "cs.AI cs.LG"
    expected[1, LABELS.index("cs.AI")] = 1.0
    expected[1, LABELS.index("cs.LG")] = 1.0

    # Third sample: "" (empty string)
    # All zeros

    # Fourth sample: "cs.LG"
    expected[3, LABELS.index("cs.LG")] = 1.0

    np.testing.assert_array_equal(result, expected)
    assert result.shape == (4, len(LABELS))
    assert result.dtype == np.float32


def test_extract_labels_batch_single_category():
    # Test batch where each input has exactly one category
    categories = ["cs.AI", "cs.LG", "cs.CV"]
    result = extract_labels(categories)

    expected = np.zeros((3, len(LABELS)), dtype=np.float32)
    expected[0, LABELS.index("cs.AI")] = 1.0
    expected[1, LABELS.index("cs.LG")] = 1.0
    expected[2, LABELS.index("cs.CV")] = 1.0

    np.testing.assert_array_equal(result, expected)
    assert np.sum(result, axis=1).tolist() == [
        1.0,
        1.0,
        1.0,
    ]  # Each row should sum to 1


def test_extract_labels_batch_empty():
    # Test with batch of empty strings
    categories = ["", "", ""]
    result = extract_labels(categories)

    expected = np.zeros((3, len(LABELS)), dtype=np.float32)

    np.testing.assert_array_equal(result, expected)
    assert np.sum(result) == 0  # All elements should be zero
