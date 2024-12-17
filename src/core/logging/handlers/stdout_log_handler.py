from logging import LogRecord, StreamHandler
from copy import copy
import logging
from typing import TextIO, override

COLOR_LEVEL_MAP = {
    'CRITICAL': "\033[1;31mCRITICAL\033[0m",
    'ERROR': "\033[1;31mERROR\033[0m",
    'WARNING': "\033[1;33mWARNING\033[0m",
    'INFO': "\033[1;32mINFO\033[0m",
    'DEBUG': "\033[1;30mDEBUG\033[0m",
    'NOTIFY': "\033[1;34mNOTIFY\033[0m"
}

COLOR_FORMAT = '\033[34m%(asctime)s\033[0m [%(levelname)s] (%(name)s|%(threadName)s) %(message)s' # noqa

class StdoutFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(COLOR_FORMAT)

    @override
    def formatMessage(self, record: LogRecord) -> str:
        record = copy(record)
        #noinspection SpellCheckingInspection
        record.levelname = COLOR_LEVEL_MAP.get(record.levelname, record.levelname)
        result = super().formatMessage(record)
        return result.replace("\n", "\n\t")

def add_handler(logger: logging.Logger, stream) -> None:
    handler = StreamHandler(stream = stream)
    handler.setFormatter(StdoutFormatter())
    logger.addHandler(handler)