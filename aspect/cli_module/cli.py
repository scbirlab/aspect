"""Command-line interface for aspect."""

from argparse import FileType
import os
import sys

from carabiner.cliutils import CLIOption, CLICommand, CLIApp

from .. import app_name, __version__
from .featurize import _featurize, _serialize

def main() -> None:

    input_file = CLIOption(
        'input_file',
        type=FileType('r'),
        default=sys.stdin,
        nargs='?',
        help='Input file. Default: STDIN',
    )
    input_filename = CLIOption(
        'input_file',
        type=str,
        help='Input file.',
    )
    feature_cols = CLIOption(
        '--features', '-x',
        type=str,
        nargs='*',
        default=None,
        help='Featurization spec: column_name[:transform[(kwargs)]:...][@output_name]...',
    )
    cache = CLIOption(
        '--cache',
        type=str,
        default=None,
        help='Where to cache data.',
    )
    _config = CLIOption(
        '--config',
        type=str,
        default=None,
        help='Load pipeline from this config or checkpoint. Default: do not use, process from scratch.',
    )
    _checkpoint = CLIOption(
        '--checkpoint',
        type=str,
        default=None,
        help='Save data at this checkpoint. Default: do not save checkpoint.',
    )

    output_name = CLIOption(
        '--output', '-o', 
        type=str,
        required=True,
        help='Output filename.',
    )

    # slice dataset
    slice_start = CLIOption(
        '--start', 
        type=int,
        default=0,
        help='First row of dataset to process.',
    )
    slice_end = CLIOption(
        '--end', 
        type=int,
        default=None,
        help='Last row of dataset to process. Default: end of dataset.',
    )
    extra_cols = CLIOption(
        '--extras',
        type=str,
        nargs="*",
        default=None,
        help='Extra columns to retain without transformation.',
    )
    random_seed = CLIOption(
        '--seed', '-e', 
        type=int,
        default=None,
        help='Random seed. Default: determininstic.',
    )

    serialize = CLICommand(
        "serialize",
        description="Checkpoint a feature spec.",
        options=[
            output_name,
            feature_cols,
            extra_cols,
        ],
        main=_serialize,
    )

    featurize = CLICommand(
        "featurize",
        description="Featurize a table.",
        options=[
            input_filename, 
            output_name,
            feature_cols,
            extra_cols,
            slice_start,
            slice_end,
            _config,
            _checkpoint,
            random_seed,
            cache,
        ],
        main=_featurize,
    )

    app = CLIApp(
        app_name, 
        description="Serializable featurization pipelines for ML/AI on chemistry, taxonomy, and general tabular data.", 
        version=__version__, 
        commands=[
            serialize,
            featurize,
        ],
    )
    app.run()
    return None


if __name__ == '__main__':
    main()
