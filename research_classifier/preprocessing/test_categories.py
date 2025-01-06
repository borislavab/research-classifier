from .categories import get_labels, category_map
from research_classifier.analysis.categories import distinct_categories
import pytest
from .categories import CategoriesProcessor
import numpy as np


@pytest.fixture(scope="module")
def processor():
    return CategoriesProcessor()


def test_get_labels_maps_old_categories():
    """Test that old category names are properly mapped to new ones"""
    labels = get_labels()

    # Check that old categories are mapped to new ones
    assert "cmp-lg" not in labels  # old category
    assert "cs.CL" in labels  # mapped to new category
    assert "supr-con" not in labels  # old category
    assert "cond-mat.supr-con" in labels  # mapped to new category


def test_get_labels_returns_sorted():
    """Test that labels are returned in sorted order"""
    labels = get_labels()
    sorted_labels = sorted(labels)
    assert labels == sorted_labels


def test_get_labels_length():
    """Test that the length is correct"""
    labels = get_labels()
    unique_labels = set(labels)
    assert len(labels) == len(unique_labels)
    assert len(labels) == len(distinct_categories) - len(category_map)


def test_extract_labels_single_category(processor):
    # Test with a single category that exists in LABELS
    categories = "cs.AI"
    result = processor.process(categories)

    # Should be all zeros except for cs.AI position
    expected = [0] * len(processor.labels)
    expected[processor.labels.index("cs.AI")] = 1

    np.testing.assert_array_equal(result, expected)
    assert sum(result) == 1  # Only one category should be marked


def test_extract_labels_old_category(processor):
    # Test with a single category that exists in LABELS
    category = "cmp-lg"
    expected_label = category_map[category]
    assert category != expected_label

    result = processor.process(category)

    # Should be all zeros except for cs.AI position
    expected = [0] * len(processor.labels)
    expected[processor.labels.index(expected_label)] = 1

    np.testing.assert_array_equal(result, expected)
    assert sum(result) == 1  # Only one category should be marked


def test_extract_labels_multiple_categories(processor):
    categories = "cs.AI cs.LG cmp-lg"
    expected_labels = ["cs.AI", "cs.LG", category_map["cmp-lg"]]
    result = processor.process(categories)

    expected = [0] * len(processor.labels)
    for label in expected_labels:
        expected[processor.labels.index(label)] = 1

    np.testing.assert_array_equal(result, expected)
    assert sum(result) == len(expected_labels)


def test_extract_labels_empty_string(processor):
    # Test with empty string
    categories = ""
    result = processor.process(categories)

    expected = [0] * len(processor.labels)

    np.testing.assert_array_equal(result, expected)
    assert sum(result) == 0  # No categories should be marked


def test_extract_labels_batch(processor):
    # Test with a batch of category strings
    categories = ["cs.AI", "cs.AI cs.LG", "", "cmp-lg"]
    result = processor.process(categories)

    # Expected results for each input
    expected = np.zeros((4, len(processor.labels)), dtype=np.float32)

    # First sample: "cs.AI"
    expected[0, processor.labels.index("cs.AI")] = 1.0

    # Second sample: "cs.AI cs.LG"
    expected[1, processor.labels.index("cs.AI")] = 1.0
    expected[1, processor.labels.index("cs.LG")] = 1.0

    # Third sample: "" (empty string)
    # All zeros

    # Fourth sample: "cmp-lg"
    expected[3, processor.labels.index(category_map["cmp-lg"])] = 1.0

    np.testing.assert_array_equal(result, expected)
    assert result.shape == (4, len(processor.labels))
    assert result.dtype == np.float32
    assert np.sum(result, axis=1).tolist() == [
        1.0,
        2.0,
        0.0,
        1.0,
    ]


def test_extract_labels_batch_empty(processor):
    # Test with batch of empty strings
    categories = ["", "", ""]
    result = processor.process(categories)

    expected = np.zeros((3, len(processor.labels)), dtype=np.float32)

    np.testing.assert_array_equal(result, expected)
    assert np.sum(result) == 0  # All elements should be zero
