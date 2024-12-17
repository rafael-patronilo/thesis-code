import unittest
import doctest

def setUpModule():
    import logging
    logging.basicConfig(level=logging.DEBUG)
#
# def tearDownModule():
#     pass

# def load_tests(loader : unittest.TestLoader, tests : unittest.TestSuite, pattern : str | None):
#     if pattern is None:
#         pattern = 'test_*.py'
#     tests.addTests(loader.discover('.core', pattern=pattern))
#     tests.addTests(doctest.DocTestSuite('core'))
#     return tests