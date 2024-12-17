"""
Provides utilities for basic framework initialization with
parsing of command line arguments and environment variables.
"""
from .global_options import GlobalOptions
from .options_parsing import register_all_options, resolve
from argparse import ArgumentParser, Namespace
import sys
from . import default_scripts
from .script_registry import Script, ScriptLoadOptions, scripts, register_from_package, argparse_register_scripts
import logging
from datetime import datetime


start_time = datetime.now()

logger = logging.getLogger()

HELP = """A custom framework for training and evaluating neural networks,
with a focus on integration with symbolic systems for explainability and 
interpretability."""

options : GlobalOptions = GlobalOptions()

register_from_package(default_scripts, scripts)

def import_torch():
    """
    Imports and configures torch by setting the preferred device.
    """
    # TODO refactor function
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
    logger.info(f"Using torch device: {device}")

def import_datasets():
    #TODO is this still needed?
    logger.info("Importing datasets")
    import datasets

def parse_args(args : list[str] | None = None, prog : str | None = None):
    """
    Parses command line arguments and returns the resolved global options and parsed args.

    :param prog: The name of the program for formatting help messages.
        If None, uses ``sys.argv[0]``
    :param args: The command line arguments to parse. If None, uses ``sys.argv[1:]``
    """
    if args is None:
        args = sys.argv[1:]
    if prog is None:
        prog = sys.argv[0]
    parser = ArgumentParser(prog=prog, description=HELP)
    register_all_options(GlobalOptions, parser)
    argparse_register_scripts(parser, scripts)
    parsed_args = parser.parse_args(args)
    return resolve(GlobalOptions, parsed_args), parsed_args

def run_script(script : Script, parsed_args : Namespace):
    """
    Runs the selected script, resolving options if required

    :param script:
    :param parsed_args:
    """
    logger.debug(f"Selected script:\n{script}")
    load_options = script.load_options or ScriptLoadOptions()
    if load_options.import_torch:
        import_torch()
    if load_options.import_datasets:
        import_datasets()
    if script.pre_load is not None:
        logger.debug("Running script pre_load")
        script.pre_load()
    if script.options_cls is not None:
        logger.debug('Resolving script options')
        script_options = resolve(script.options_cls, parsed_args)  # type: ignore
        logger.info(f"Entering script {script.name} with options:\n{script_options}")
        script.main(script_options)  # type: ignore
    else:
        logger.info(f"Entering script {script.name}")
        script.main()  # type: ignore

__all__ = [
    'options',
    'start_time',
    'import_torch',
    'parse_args',
    'run_script',
    'import_datasets'
]