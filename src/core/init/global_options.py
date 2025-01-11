from dataclasses import dataclass, field
import re
from .options_parsing import option, comma_key_values
from pathlib import Path
from core.util.strings import multiline_repr


@dataclass
class GlobalOptions:
    """
    Global options for the framework.
    """

    #logging options
    log_level: str = field(
        default='INFO',
        metadata=option(str, help_=
        'Logging level. Defaults to INFO')
    )
    log_handler_level : dict[str, str] = field(
        default_factory=lambda:{
            'file' : 'DEBUG'
        },
        metadata=option(comma_key_values, help_=
        'Comma separated key value pairs indicating specific logging levels for handlers. '
        'Available handlers are discord, console and file. '
        'Example: --log-handler-level console=INFO,file=DEBUG')
    )
    log_dir: Path = field(
        default=Path("logs"),
        metadata=option(Path, help_=
        'Directory to store logs in')
    )
    discord_webhook_url: str = field(
        default="",
        metadata=option(str, help_=
        'Discord webhook URL. If not specified or empty no messages '
        'will be sent to discord.')
    )
    discord_level_map: dict[str, str] = field(
        default_factory=dict,
        metadata=option(comma_key_values, help_=
        'Comma separated key value pairs indicating how to map log '
        'levels to discord mentions.\n'
        'Example: --discord-level-map INFO=INFO@here,ERROR=ERROR@everyone')
    )

    # Paths
    models_path: Path = field(
        default=Path("storage/models"),
        metadata=option(Path, help_=
        'Path to store and load models from')
    )
    studies_path: Path = field(
        default=Path("storage/studies"),
        metadata=option(Path, help_=
        'Path to store and load studies from')
    )
    data_path: Path = field(
        default=Path("data"),
        metadata=option(Path, help_=
        'Path to store and load datasets from')
    )
    debug_path: Path = field(
        default=Path("debug")
        , metadata=option(Path, help_=
        'Path to store debug data at')
    )


    num_loaders: int = field(
        default=4,
        metadata=option(int, help_=
        'Number of data loading processes to use')
    )

    preferred_device: str = field(
        default='cuda',
        metadata=option(str, help_=
        "Preferred torch device to use. "
        "The 'force:' prefix will cause an error if the device is not available.")
    )

    def __repr__(self):
        discord_webhook_url = 'None'
        if len(self.discord_webhook_url) > 0:
            discord_webhook_url = '[hidden]'
        return multiline_repr(self, discord_webhook_url=discord_webhook_url)