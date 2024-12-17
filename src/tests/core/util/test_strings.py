import unittest
from core.util.strings import *

class MyTestCase(unittest.TestCase):
    def test_truncate(self):
        self.assertEqual(
            truncate_string(
                "This is a test string",
                10),
            "This is a ")
        self.assertEqual(
            truncate_string(
                "This is a test string",
                10,
                "..."),
            "This is...")
        self.assertEqual(
            truncate_string(
                "This is a test string",
                10,
                "...{chars}"),
            "This ...16")
        self.assertEqual(
            truncate_string(
                "This is a test string",
                10,
                "{chars}..."),
            "This 16...")
        self.assertEqual(
            truncate_string(
                "This is a test string",
                1,
            "...{chars}"),
            ".")

if __name__ == '__main__':
    unittest.main()
