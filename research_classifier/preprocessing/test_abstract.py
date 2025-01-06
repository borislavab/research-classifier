import pytest
from .abstract import StopWordRemover


@pytest.fixture
def processor():
    return StopWordRemover()


def test_basic_stop_word_removal(processor):
    text = "this is a test sentence"
    expected = "test sentence"
    assert processor.process(text) == expected


def test_empty_string(processor):
    text = ""
    expected = ""
    assert processor.process(text) == expected


def test_only_stop_words(processor):
    text = "the and is at"
    expected = ""
    assert processor.process(text) == expected


def test_no_stop_words(processor):
    text = "cat dog house tree"
    expected = "cat dog house tree"
    assert processor.process(text) == expected


def test_multiple_spaces(processor):
    text = "this   is   a    test"
    expected = "test"
    assert processor.process(text) == expected


def test_multi_line_multi_sentence(processor):
    text = "this is a test sentence.\r\n   this is another test sentence. \r\n\r\n"
    expected = "test sentence. another test sentence."
    assert processor.process(text) == expected


def test_mixed_case(processor):
    text = "The CAT is ON the MAT"
    expected = "CAT MAT"
    assert processor.process(text) == expected
