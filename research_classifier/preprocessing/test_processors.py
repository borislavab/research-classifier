import unittest
from .processors import StopWordRemover


class TestStopWordRemover(unittest.TestCase):
    def setUp(self):
        self.processor = StopWordRemover()

    def test_basic_stop_word_removal(self):
        text = "this is a test sentence"
        expected = "test sentence"
        self.assertEqual(self.processor.process(text), expected)

    def test_empty_string(self):
        text = ""
        expected = ""
        self.assertEqual(self.processor.process(text), expected)

    def test_only_stop_words(self):
        text = "the and is at"
        expected = ""
        self.assertEqual(self.processor.process(text), expected)

    def test_no_stop_words(self):
        text = "cat dog house tree"
        expected = "cat dog house tree"
        self.assertEqual(self.processor.process(text), expected)

    def test_multiple_spaces(self):
        text = "this   is   a    test"
        expected = "test"
        self.assertEqual(self.processor.process(text), expected)

    def test_multi_line_multi_sentence(self):
        text = "this is a test sentence.\r\n   this is another test sentence. \r\n\r\n"
        expected = "test sentence. another test sentence."
        self.assertEqual(self.processor.process(text), expected)

    def test_mixed_case(self):
        text = "The CAT is ON the MAT"
        expected = "CAT MAT"
        self.assertEqual(self.processor.process(text), expected)
