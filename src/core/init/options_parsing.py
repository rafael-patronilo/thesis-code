"""
Utilities for defining options dataclasses with automatic assignment from
command line arguments and environment variables.
"""
import dataclasses
import warnings
from typing import Callable, Iterable, Mapping, Any, TypedDict, overload
from argparse import ArgumentParser, Namespace
import os


def to_arg_casing(python_symbol : str, is_positional : bool) -> str:
    """
    Converts a python symbol to a command line argument key, that is
    ``lower_snake_case`` to ``kebab-case``,
    prepending with '--' for non-positional arguments


    :param python_symbol: The python symbol to convert
    :param is_positional: Whether this argument is positional

    :return:
    """
    arg_key = python_symbol.replace('_', '-')
    if not is_positional:
        arg_key = '--' + arg_key
    return arg_key

def to_env_casing(python_symbol : str) -> str:
    """
    Converts a python symbol to an environment variable key, that is
    ``lower_snake_case`` to ``UPPER_SNAKE_CASE``


    :param python_symbol: The python symbol to convert

    :return:
    """
    return python_symbol.upper()

class INFER: # noqa
    pass

class NO_VALUE: # noqa
    pass


class Option[T]:
    """
    Class to handle parsing for option fields.
    It is assigned as metadata to fields in dataclasses, under the key .
    """
    def __init__(self,
                 flags_or_name: list[str | INFER],
                 env_key : str | None | INFER,
                 is_positional : bool,
                 required : bool,
                 help_ : str,
                 from_string : Callable[[str], T]
                 ):
        self.flags_or_name = flags_or_name
        self.env_key = env_key
        self.is_positional = is_positional
        self.required = required
        self.help_ = help_
        self.from_string = from_string

    def argparse_register(self, parser : ArgumentParser, field : dataclasses.Field) -> None:
        """
        Registers the option with an argument parser

        :param parser: The parser to register the option with
        :param field: The field for this option
        :return:
        """
        if len(self.flags_or_name) == 0:
            return
        flags_or_name : list[str] = [
            flag if isinstance(flag, str) else to_arg_casing(field.name, self.is_positional)
            for flag in self.flags_or_name
        ]
        if self.is_positional and len(flags_or_name) != 1:
            raise ValueError('Positional arguments must have only one name')
        assert not any(flag is INFER for flag in flags_or_name)
        if self.is_positional:
            parser.add_argument(
                flags_or_name[0],
                help=self.help_,
                dest=field.name
            )
        else:
            parser.add_argument(
                *flags_or_name,
                required=self.required,
                help=self.help_,
                dest=field.name
            )

    def resolve_arg(self, parsed_args : Namespace, field : dataclasses.Field) -> T | NO_VALUE:
        """
        Resolves the value of the option from parsed arguments

        :param parsed_args: The parsed arguments
        :param field: The field for this option

        :return: The resolved value or NO_VALUE if not found
        """
        value = vars(parsed_args).get(field.name)
        if value is None:
            return NO_VALUE()
        else:
            return self.from_string(value)

    def resolve_env(self, environ : Mapping[str, Any], field : dataclasses.Field) -> T | NO_VALUE:
        """
        Resolves the value of the option from environment variables

        :param environ: The environment variables
        :param field: The field for this option

        :return: The resolved value or NO_VALUE if not found
        """
        env_key = self.env_key
        if env_key is None:
            return NO_VALUE()
        elif isinstance(env_key, INFER):
            env_key = to_env_casing(field.name)
        value = environ.get(env_key, None)
        if value is None:
            return NO_VALUE()
        else:
            return self.from_string(value)

    def resolve(
            self,
            parsed_args : Namespace,
            default_factory : Callable[[], T] | None,
            field : dataclasses.Field,
            environ : Mapping[str, Any] = os.environ) -> T:
        """
        Resolves the value of the option, first checking parsed arguments, then
        environment variables, and finally the default factory.

        :param parsed_args: The parsed arguments, first to be checked
        :param default_factory: The default factory for the field, last to be checked
        :param field: The option's field
        :param environ: Mapping of environment variables, second to be checked.
            Defaults to ``os.environ``

        :return: The resolved value
        :raises ValueError: If no value is found
        """
        resolvers : list[Callable[[], T | NO_VALUE]] = [
            lambda: self.resolve_arg(parsed_args, field),
            lambda: self.resolve_env(environ, field)
        ]
        if default_factory is not None:
            resolvers.append(default_factory)
        for resolver in resolvers:
            value = resolver()
            if not isinstance(value, NO_VALUE):
                return value
        raise ValueError(f'Value missing for field {field.name}')

def register_all_options(cls : type, parser : ArgumentParser):
    """
    Registers all options in a dataclass with an argument parser
    :param parser: The parse where the options should be registered
    :param cls: The dataclass with the options
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f'Expected class {cls} to be a dataclass')
    fields : Iterable[dataclasses.Field] = dataclasses.fields(cls) # noqa
    for field in fields:
        option_object : Option | None = field.metadata.get('option')
        if option_object is None:
            warnings.warn(f'Field {field.name} is missing option metadata; Ignoring field')
            continue
        option_object.argparse_register(parser, field)

def resolve[T](cls : type[T], parsed_args : Namespace, environ : Mapping[str, Any] = os.environ) -> T:
    """
    Resolves all options in a dataclass from parsed arguments

    :param environ: mapping that serves and environment. Defaults to ``os.environ``
    :param cls: The dataclass with the options
    :param parsed_args: The parsed arguments

    :return: An instance of the dataclass with the resolved options
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f'Expected class {cls} to be a dataclass')
    resolved_args = {}
    for field in dataclasses.fields(cls): # type: ignore
        option_object : Option | None = field.metadata.get('option')
        if option_object is None:
            warnings.warn(f'Field {field.name} is missing option metadata; Ignoring field')
            continue
        default_factory = None
        if field.default is not dataclasses.MISSING:
            default_factory = lambda : field.default
        elif field.default_factory is not dataclasses.MISSING:
            default_factory = field.default_factory
        resolved_args[field.name] = option_object.resolve(parsed_args, default_factory, field, environ)
    return cls(**resolved_args)


def option[T](
        from_string: Callable[[str], T],
        *flags: str | INFER | None,
        env_key : str | None | INFER = INFER(),
        help_ : str = "",
) -> Mapping:
    """
    Defines an option which can be changed with command line arguments
    or environment variables.
    If both are specified, the command line argument takes precedence.

    When the INFER sentinel object is used instead for the environment key or flags,
    the value will be inferred from the field name.

    >>> from dataclasses import dataclass, field
    >>> from argparse import ArgumentParser
    >>>
    >>> @dataclass()
    >>> class MyOptions:
    ...     my_option : int = field(1, metadata=option(int))
    >>>
    >>> parser = ArgumentParser()
    >>> register_all_options(MyOptions, parser)
    >>> resolve(MyOptions, parser.parse_args(['--my-option', '2']))
    MyOptions(my_option=2)
    >>> resolve(MyOptions, parsed_args=parser.parse_args([]), environ={'MY_OPTION': '3'})
    MyOptions(my_option=3)
    >>>

    :param flags: either None to indicate and environment variable only option,
        no argument to indicate INFER or
        one more flags for this option.
        If INFER is among the flags, it will be replaced by the name inferred from the field name.
    :param env_key: Either None to not read environment variables,
        INFER to infer the key from the field name or the key to use.
    :param from_string: A function to convert a string to the option type.
        If None, will use the constructor inferred from default
    :param help_: The help text for this option

    :return: The option field metadata
    """
    flag_list : list[str | INFER]
    if None in flags:
        flag_list = []
    elif len(flags) == 0:
        flag_list = [INFER(),]
    else:
        flag_list = list(flags) #type: ignore
    assert None not in flags
    if sum(1 for flag in flags if flag is INFER) > 1:
        raise ValueError('Only one INFER flag is allowed')
    assert from_string is not None
    metadata = dict(
        option = Option(flag_list, env_key = env_key, is_positional=False,
                                 required = False, help_ = help_,
                                 from_string = from_string)
    )
    return metadata

def positional[T](
        from_string : Callable[[str], T],
        arg_name : str | INFER = INFER(),
        help_ : str = ""
) -> T:
    """
    Defines a required positional command line argument


    :param help_: The help text for this option
    :param from_string: A function to convert a string to the option type.
    :param arg_name: The name of the argument. If INFER, the name will be inferred from the field name as
        described for option.

    :return: A dataclass field with the option metadata
    """
    metadata = { 'option': Option[T]([arg_name], env_key = None,
                                     required = True, help_ = help_,
                                     is_positional=True,
                                     from_string = from_string)}
    return dataclasses.field(metadata=metadata)

def comma_key_values[T](arg : str, from_string : Callable[[str], T] = lambda x:x ) -> dict[str, T]:
    """
    Parses a comma separated list of key value pairs

    :param from_string: An optional conversion function for the values
    :param arg: The string to parse

    :return: A dictionary of key value pairs
    """
    key_values = arg.split(',')
    key_values = (kv.split('=') for kv in key_values)
    return {key : from_string(value) for key, value in key_values}
