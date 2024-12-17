import logging
from core.init import *
import core.init
from core.init.script_registry import Script
from core.logging import setup as setup_logging
import logging

def main(args : list[str] | None = None, prog : str | None = None):
    """
    Loads the framework, selecting a script from the command line arguments.

    :param prog: The name of the program for formatting help messages.
        If None, uses ``sys.argv[0]``
    :param args: The command line arguments to parse. If None, uses ``sys.argv[1:]``
    """
    core.init.options, parsed_args = parse_args(args, prog)

    setup_logging()

    logger = logging.getLogger()

    logger.info(f"Loaded options:\n{core.init.options}")
    script : Script | None = vars(parsed_args).get('script')
    if script is None:
        logger.warning("No script specified, run with --help for script list.\nExiting.")
        return
    else:
        run_script(script, parsed_args)