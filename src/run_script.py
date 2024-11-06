#!/usr/bin/env python
import command_base
import sys
from core.util import is_import_safe
import importlib


@command_base.main_wrapper
def main():
    # Resume training from the last checkpoint
    script_path = sys.argv[1]
    sys.argv = sys.argv[1:]
    assert is_import_safe(script_path), f"Invalid script name: {script_path}"
    script = importlib.import_module('.' + script_path, 'scripts')
    script.main()

if __name__ == '__main__':
    main()
