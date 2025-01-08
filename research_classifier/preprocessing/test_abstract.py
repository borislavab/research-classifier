import pytest
from .abstract import StopWordRemover, Lemmatizer


@pytest.fixture
def stop_word_remover():
    return StopWordRemover()


@pytest.fixture
def lemmatizer():
    return Lemmatizer()


def test_stop_words_basic(stop_word_remover):
    text = "this is a test sentence"
    expected = "test sentence"
    assert stop_word_remover.process(text) == expected


def test_stop_words_empty_string(stop_word_remover):
    text = ""
    expected = ""
    assert stop_word_remover.process(text) == expected


def test_stop_words_only_stop_words(stop_word_remover):
    text = "the and is at"
    expected = ""
    assert stop_word_remover.process(text) == expected


def test_stop_words_no_stop_words(stop_word_remover):
    text = "cat dog house tree"
    expected = "cat dog house tree"
    assert stop_word_remover.process(text) == expected


def test_stop_words_multiple_spaces(stop_word_remover):
    text = "this   is   a    test"
    expected = "test"
    assert stop_word_remover.process(text) == expected


def test_stop_words_multi_line_multi_sentence(stop_word_remover):
    text = "this is a test sentence.\r\n   this is another test sentence. \r\n\r\n"
    expected = "test sentence. another test sentence."
    assert stop_word_remover.process(text) == expected


def test_stop_words_mixed_case(stop_word_remover):
    text = "The CAT is ON the MAT"
    expected = "CAT MAT"
    assert stop_word_remover.process(text) == expected


def test_lemmatizer_empty_string(lemmatizer):
    result = lemmatizer.process("")
    assert result == ""


def test_lemmatizer_single_word(lemmatizer):
    result = lemmatizer.process("dogs")
    assert result == "dog"


def test_lemmatizer_multiple_words(lemmatizer):
    result = lemmatizer.process("dogs and corpora")
    assert result == "dog and corpus"


def test_lemmatizer_multiple_spaces(lemmatizer):
    result = lemmatizer.process("this   is   a    test")
    assert result == "this is a test"


def test_lemmatizer_multi_line_multi_sentence(lemmatizer):
    result = lemmatizer.process(
        "this is a test sentence.\r\n  another test sentence. \r\n\r\n"
    )
    assert result == "this is a test sentence. another test sentence."


def test_lemmatizer_special_characters(lemmatizer):
    result = lemmatizer.process("this is a test sentence. t#2 -> ** test sentence.")
    assert result == "this is a test sentence. t#2 -> ** test sentence."


def test_lemmatizer_mixed_case(lemmatizer):
    result = lemmatizer.process("The CAT is ON the MAT")
    assert result == "The CAT is ON the MAT"
