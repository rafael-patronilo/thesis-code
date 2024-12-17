from ast import Call
import logging
from copy import deepcopy, copy
from os import truncate
from typing import Callable
from functools import wraps
from core.util.strings import truncate_string

class MultiLineFormatter(logging.Formatter):
    """
    Wraps a formatter to add a prefix (default tab)
        on every additional line of a log message.
    """
    def __init__(self, formatter : logging.Formatter, prefix = "\t"):
        super().__init__()
        self.formatter = formatter
        self.prefix = prefix

    def format(self, record):
        message = self.formatter.format(record)
        return message.replace("\n", "\n" + self.prefix)

