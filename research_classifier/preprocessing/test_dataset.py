from .dataset import extract_labels, LABELS


def test_extract_labels_single_category():
    # Test with a single category that exists in LABELS
    categories = "cs.AI"
    result = extract_labels(categories)

    # Should be all zeros except for cs.AI position
    expected = [0] * len(LABELS)
    expected[LABELS.index("cs.AI")] = 1

    assert result == expected
    assert sum(result) == 1  # Only one category should be marked


def test_extract_labels_multiple_categories():
    # Test with multiple valid categories
    categories = "cs.AI cs.LG"
    result = extract_labels(categories)

    expected = [0] * len(LABELS)
    expected[LABELS.index("cs.AI")] = 1
    expected[LABELS.index("cs.LG")] = 1

    assert result == expected
    assert sum(result) == 2  # Two categories should be marked


def test_extract_labels_empty_string():
    # Test with empty string
    categories = ""
    result = extract_labels(categories)

    expected = [0] * len(LABELS)

    assert result == expected
    assert sum(result) == 0  # No categories should be marked
