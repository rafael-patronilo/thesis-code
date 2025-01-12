import doctest
import unittest

import core.init.options_parsing
from core.init.options_parsing import option, register_all_options, resolve
from dataclasses import dataclass, field
from argparse import ArgumentParser

class TestOptionsParsing(unittest.TestCase):

    def test_option_infer(self):
        @dataclass
        class MyOptions:
            my_option : int = field(default=1, metadata=option(int))

        parser = ArgumentParser()
        register_all_options(MyOptions, parser)
        result = resolve(MyOptions, parser.parse_args(['--my-option', '2']), {})
        self.assertEqual(result.my_option, 2)
        result = resolve(MyOptions, parsed_args=parser.parse_args([]), config_dict={}, environ={'MY_OPTION': '3'})
        self.assertEqual(result.my_option, 3)
        #TODO test config dict


def load_tests(loader : unittest.TestLoader, tests : unittest.TestSuite, pattern : str | None):
    #tests.addTests(doctest.DocTestSuite(core.init.options_parsing))
    return tests