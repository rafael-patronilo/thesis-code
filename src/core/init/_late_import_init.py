"""
Required to avoid cyclic import with core.logging
"""

import logging
from datetime import datetime
from core.init import *
import core.init
from core.init.scripts import Script
from core.logging import NOTIFY, setup as setup_logging
import logging

logger = logging.getLogger()

def main(args : list[str] | None = None, prog : str | None = None):
    """
    Loads the framework, selecting a script from the command line arguments.

    :param prog: The name of the program for formatting help messages.
        If None, uses ``sys.argv[0]``
    :param args: The command line arguments to parse. If None, uses ``sys.argv[1:]``
    """
    core.init.options, parsed_args, config_files, script_configs = parse_args(args, prog)

    setup_logging()
    if len(config_files) > 0:
        logger.info(f"Configurations files:\n{'\n'.join(f'\t{f}' for f in config_files)}")
    logger.info(f"Loaded options:\n{core.init.options}")
    script : Script | None = vars(parsed_args).get('script')
    if script is None:
        logger.warning("No script specified, run with --help for script list.\nExiting.")
        return
    else:
        run_script(script, parsed_args, script_configs)
    logger.log(NOTIFY, f"Script complete in {datetime.now() - start_time}")
    logging.shutdown()