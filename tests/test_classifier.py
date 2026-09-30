import unittest
from src.preprocess import clean_text

class TestClassifierUtilities(unittest.TestCase):
    def test_clean_text_lowercases(self):
        result = clean_text("FREE OFFER NOW!!!")
        self.assertEqual(result, "free offer now")

    def test_clean_text_replaces_url(self):
        result = clean_text("Visit https://example.com now")
        self.assertNotIn("https", result)
        self.assertNotIn("example.com", result)

    def test_clean_text_returns_string(self):
        self.assertIsInstance(clean_text("Hello world"), str)

if __name__ == "__main__":
    unittest.main()
