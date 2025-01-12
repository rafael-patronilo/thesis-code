"""
Utilities for managing scripts.
"""
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import NamedTuple, Callable, Any, Self, Type
from dataclasses import dataclass
from .options_parsing import register_all_options
from collections import OrderedDict
import pkgutil
import importlib
import inspect
from core.util.strings import multiline_repr

class ScriptLoadOptions(NamedTuple):
    """
    Options for loading scripts.

    :param import_torch: Whether to import and initialize torch.
    :param import_datasets: Whether to import the module ``datasets``.
    """
    import_torch : bool = True
    import_datasets : bool = True

@dataclass
class Script:
    """
    Dataclass representing a script.

    :param options_cls: The options class for the script.
    :param pre_load: A function to run before main function but after
        the framework initialization.
        Modules will be imported during script discovery, therefore
        the script should import large modules, such as torch, here.
    :param main: The main function of the script.
        If an options class is provided, options will be parsed, and
        it will be passed as an argument.
    :param name: The name of the script.
    :param help_str: The help string for the script.
    :param script_path: The path to the script file.
    :param load_options: Options for loading the script.
    """
    options_cls: Type | None
    load_options : ScriptLoadOptions | None
    module : Any
    name: str
    fullname : str
    help_str: str
    script_path: str

    def __repr__(self):
        return multiline_repr(self)


class ScriptGroup(NamedTuple):
    """
    Named tuple representing a group of scripts.

    :param children: A mapping of script names to scripts.
    :param name: The name of the group.
    :param help_str: The help string for the group.
    """
    children: dict[str, Self | Script]
    name: str | None
    help_str: str | None

    def register_from_module(self, module):
        """
        Converts a module into a script and adds it to the registry.

        Module should define a function ``main`` with no args or with an options arg
        if an options class is defined.

        It can optionally define:
        * a function ``pre_load`` with no args which will be
            called after the framework is initialized but before the main function.
            Large modules should be imported here to avoid unnecessary imports
            during script discovery.
        * an ``Options`` dataclass using the field descriptors from
            _`core.init.options_parsing`. If present, it will be added to the parser
            and the resulting instance will be passed to the main function.
        * a ``LOAD_OPTIONS`` constant of type ``ScriptLoadOptions`` to customize
            how the script should be loaded

        Help string for the script is extracted from main function's docstring
        and the script name will be the same as the module's name



        :param module: The module to register scripts from.
        """
        if hasattr(module, 'main'):
            main = module.main
            pre_load = getattr(module, 'pre_load', None)
            load_options = getattr(module, 'LOAD_OPTIONS', None)
            if load_options is not None and not isinstance(load_options, ScriptLoadOptions):
                if isinstance(load_options, dict):
                    load_options = ScriptLoadOptions(**load_options)
                else:
                    raise ValueError(f"{module}.LOAD_OPTIONS should be a ScriptLoadOptions")
            options_cls = getattr(module, 'Options', None)
            main_signature = inspect.signature(main)
            if len(main_signature.parameters) > 1:
                raise ValueError(f"{module}.main should have no more than 1 argument")
            elif options_cls is None and len(main_signature.parameters) != 0:
                raise ValueError(f"{module}.main should has an argument but Options was not defined")

            fullname = module.__name__
            name = fullname.split('.')[-1]
            help_str = main.__doc__ if main.__doc__ is not None else ""
            script_path = module.__file__
            script = Script(options_cls, load_options, module,
                            name, fullname, help_str, script_path)
            self.children[name] = script
        else:
            raise ValueError(f"Module {module} does not define a main function")

scripts: ScriptGroup = ScriptGroup(OrderedDict(), None, None)


def register_from_package(module, out: ScriptGroup | None = None):
    """
    Recursively registers all modules in a package in a script group tree.

    Package should contain modules defining scripts as described in
    _`register_from_module`. The script group will be named after the package.

    :param module: The package to register scripts from.
    :param out: The script group to add the scripts to.
        If unspecified will use user scripts
    """
    if out is None:
        out = scripts
    sub_modules = [
        (name, importlib.import_module(name), is_pkg)
        for _, name, is_pkg in
        pkgutil.iter_modules(module.__path__, module.__name__ + '.')
    ]

    for name, submodule, is_pkg in sub_modules:
        name = name.split('.')[-1]
        if is_pkg:
            group = ScriptGroup({}, name, submodule.__doc__)
            register_from_package(submodule, group)
            out.children[name] = group
        if not is_pkg:
            out.register_from_module(submodule)

def argparse_register_scripts(parser : ArgumentParser, script_group : ScriptGroup):
    """
    Registers a script subparsers with an argument parser.
    Each script or script group will be registered as a subcommand.
    Will recurse into script groups to register
    inner scripts as sub-subcommands and so on.

    :param parser: The argument parser to add the scripts to.
    :param title: The title for the script subparsers.
    :param script_group: The script subparsers to register.
    """
    subparsers = parser.add_subparsers(title='Available scripts', metavar='SCRIPT', required=True)
    for name, script in script_group.children.items():
        subparser = subparsers.add_parser(name, help=script.help_str, description=script.help_str)
        if isinstance(script, Script):
            if script.options_cls is not None:
                register_all_options(script.options_cls, subparser)
            subparser.set_defaults(script=script)
        else:
            argparse_register_scripts(subparser, script)