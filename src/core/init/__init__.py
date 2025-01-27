"""
Provides utilities for basic framework initialization with
parsing of command line arguments and environment variables.
"""
import importlib
import os
from pathlib import Path, PosixPath
from typing import Any, Type

from .global_options import GlobalOptions
from .options_parsing import register_all_options, resolve
from argparse import ArgumentParser, Namespace
import sys
from . import default_scripts
from .scripts import Script, ScriptLoadOptions, scripts, register_from_package, argparse_register_scripts
import logging
from datetime import datetime
import json

DO_SCRIPT_IMPORTS : bool = False


start_time = datetime.now()

logger = logging.getLogger()

DEFAULT_CONFIG_FILE = 'config.json'
CONFIG_ENV_VAR = 'CONFIG_FILE'

HELP = """A custom framework for training and evaluating neural networks,
with a focus on integration with symbolic systems for explainability and 
interpretability.

For help on a specific script, run the script with the --help flag.

Options can be specified in multiple sources, with the following order of precedence:
- command line arguments
- environment variables
- JSON configuration files
- default values
Script positional arguments can only be specified in command line arguments.

Note that the option name in these different sources follows the respective naming convention. 
For example the option `--log-level` in command line arguments would be 
`LOG_LEVEL` in environment variables and log_level in JSON configuration files.

JSON configuration files can be specified with the --config-file flag 
or with the environment variable CONFIG_FILE. Multiple files can be specified,
with later files overriding options from earlier files. Command line config files 
override options from CONFIG_FILE.

If no configuration file is specified and './config.json' exists, it will be loaded by default. 
 
"""

options : GlobalOptions = None # type: ignore

register_from_package(default_scripts, scripts)

_torch_imported = False
def import_torch():
    """
    Imports and configures torch by setting the preferred device.
    """
    global _torch_imported
    if _torch_imported:
        return
    preferred_device = options.preferred_device
    force_device = False
    if preferred_device.startswith("force "):
        preferred_device = preferred_device[len("force "):]
        force_device = True

    # import torch and set device
    logger.info("Importing torch and setting device")
    import torch
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and preferred_device == "cuda"
        else "cpu"
    )
    if force_device and device.type != preferred_device:
        raise RuntimeError(f"Failed to force device to {preferred_device}")
    torch.set_default_device(device)
    torch.serialization.add_safe_globals([PosixPath]) # Added to support a bug in older checkpoints
    logger.info(f"Using torch device: {device}")
    logger.info(f"Using torch version: {torch.__version__}")
    logger.info(f"Default torch dtype: {torch.get_default_dtype()}")
    _torch_imported = True

def import_datasets():
    #TODO is this still needed?
    logger.info("Importing datasets")
    import datasets

def parse_config_files(config_files) -> tuple[dict, dict]:
    """
    Parses configuration files and returns the resolved global options and parsed args.

    :param parsed_args: The parsed command line arguments.
    """
    global_config = {}
    script_configs = {}
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config : dict = json.load(f)
            script_config = config.pop('script_options', {})
            global_config.update(config)
            for k, v in script_config.items():
                script_configs.setdefault(k, {}).update(v)
    return global_config, script_configs


def parse_args(args : list[str] | None = None, prog : str | None = None) -> tuple[GlobalOptions, Namespace, list[str], dict[str, dict]]:
    """
    Parses command line arguments and returns the resolved global options and parsed args.

    :param prog: The name of the program for formatting help messages.
        If None, uses ``sys.argv[0]``
    :param args: The command line arguments to parse. If None, uses ``sys.argv[1:]``

    :return: A tuple of the resolved global options, parsed arguments,
        list of parsed config files, and script config dicts.
    """
    if args is None:
        args = sys.argv[1:]
    if prog is None:
        prog = sys.argv[0]
    parser = ArgumentParser(prog=prog, description=HELP)
    parser.add_argument('--config-file',
                        dest='config_files', action='append', default=[],
                        help='Path to a configuration file to load. Can be specified multiple times.'
                             'Options are loaded from left to right, with later files overriding earlier ones.')
    register_all_options(GlobalOptions, parser)
    argparse_register_scripts(parser, scripts)
    parsed_args = parser.parse_args(args)
    config_files : list[str] = parsed_args.config_files
    env_config = os.getenv(CONFIG_ENV_VAR)
    if env_config is not None and Path(env_config).is_file():
        config_files.insert(0, env_config)
    if len(config_files) == 0 and Path('config.json').is_file():
        config_files.append('config.json')
    global_config, script_configs = parse_config_files(config_files)
    return resolve(GlobalOptions, parsed_args, global_config), parsed_args, config_files, script_configs

def run_script(script : Script, parsed_args : Namespace, script_configs : dict[str, dict]):
    """
    Runs the selected script, resolving options if required

    :param script_configs: The script configurations extracted from config files,
        as a mapping from full script name to script options.
    :param script: The script to be run
    :param parsed_args: The parsed command line arguments
    """
    logger.debug(f"Selected script:\n{script}")
    load_options = script.load_options or ScriptLoadOptions()
    if load_options.import_torch:
        import_torch()
    if load_options.import_datasets:
        import_datasets()
    global DO_SCRIPT_IMPORTS
    logger.debug("Reimporting script module")
    DO_SCRIPT_IMPORTS = True
    importlib.reload(script.module)
    if script.options_cls is not None:
        logger.debug('Resolving script options')
        script_options = resolve(
            script.options_cls, #type: ignore
            parsed_args, script_configs.get(script.fullname, {}))
        logger.info(f"Entering script {script.name} with options:\n{script_options}")
        script.module.main(script_options)  # type: ignore
    else:
        logger.info(f"Entering script {script.name}")
        script.module.main()  # type: ignore

def main(args : list[str] | None = None, prog : str | None = None):
    from ._late_import_init import main as real_main
    return real_main(args, prog)

__all__ = [
    'main',
    'run_script',
    'options',
    'start_time',
    'import_torch',
    'parse_args',
    'import_datasets',
    'DO_SCRIPT_IMPORTS'
]