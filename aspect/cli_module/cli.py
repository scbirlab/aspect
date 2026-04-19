"""Command-line interface for duvidnn."""

from argparse import FileType
import os
import sys

from carabiner.cliutils import CLIOption, CLICommand, CLIApp

from .. import app_name, __version__
from ..checkpoint_utils import _load_json, save_json
from ..utils.package_data import _get_data_path

from .hyperprep import _hyperprep
from .percentile import _percentile
from .predict import _predict
from .split import _split
from .train import _train

cache_dir, modelbox_name_file = _get_data_path("modelbox-names.json")
if not os.path.exists(modelbox_name_file):
    from ..autoclass import AutoModelBox
    from ..base.modelbox_registry import DEFAULT_MODELBOX, MODELBOX_NAMES
    save_json([DEFAULT_MODELBOX, MODELBOX_NAMES], modelbox_name_file)
else:
    DEFAULT_MODELBOX, MODELBOX_NAMES = _load_json(cache_dir, os.path.basename(modelbox_name_file))

_LR_DEFAULT: float = .01


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
    train_data = CLIOption(
        '--training', '-1',
        type=str,
        default=None,
        help='Training dataset. Required if no checkpoint provided.',
    )
    val_data = CLIOption(
        '--validation', '-2',
        type=str,
        required=True,
        help='Validation dataset file.',
    )
    test_data = CLIOption(
        '--test', '-t',
        type=str,
        default=None,
        help='Test dataset file.',
    )
    feature_cols = CLIOption(
        '--features', '-x',
        type=str,
        nargs='*',
        default=None,
        help='Column names from data file that contain features. Required if no checkpoint provided.',
    )
    feature_cols2 = CLIOption(
        '--x2',
        type=str,
        nargs='*',
        default=None,
        help='Column names from data file that contain interacting features.',
    )
    context = CLIOption(
        '--context',
        type=str,
        nargs='*',
        default=None,
        help='Column names from data file that contain context features.',
    )
    structure_col = CLIOption(
        '--structure', '-S',
        type=str,
        default=None,
        help="""
        Column names from data file that contains a string representation 
        of chemical structure. Required for chemical model classes.
        """,
    )
    structure_representation = CLIOption(
        '--input-representation', '-R',
        type=str,
        default="smiles",
        choices=["smiles", "selfies", "inchi", "aa_seq"],
        help='Type of chemical structure string. Default: SMILES if training; for prediction, use same as training data.',
    )
    label_cols = CLIOption(
        '--labels', '-y',
        type=str,
        nargs='*',
        default=None,
        help='Column names from data file that contain labels. Required if no checkpoint provided.',
    )
    cache = CLIOption(
        '--cache',
        type=str,
        default=".",
        help='Where to cache data.',
    )
    model_class = CLIOption(
        '--model-class', '-k',
        type=str,
        default=DEFAULT_MODELBOX,
        choices=MODELBOX_NAMES,
        help='Test dataset file.',
    )
    fusion_method = CLIOption(
        '--fusion',
        type=str,
        default="product",
        choices=["product", "sum", "concat"],
        help='Method for fusing bilinear model.',
    )
    _checkpoint = CLIOption(
        '--checkpoint',
        type=str,
        default=None,
        help='Load a modelbox from this checkpoint. Default: do not use, make a new modelbox.',
    )
    n_units = CLIOption(
        '--units', '-u',
        type=int,
        default=8,
        help='Number of units per hidden layer.',
    )
    n_hidden = CLIOption(
        '--hidden', '-m',
        type=int,
        default=1,
        help='Number of hidden layers.',
    )
    residual_depth = CLIOption(
        '--residual',
        type=int,
        default=None,
        help='Depth of residual blocks. Default: Do not use residual blocks.',
    )
    _2d = CLIOption(
        '--descriptors', 
        action="store_true",
        help='Use 2d descriptors (needs a SMILES input feature).',
    )
    _fp = CLIOption(
        '--fp', 
        action="store_true",
        help='Use chemical fingerprints (needs a SMILES input feature).',
    )
    dropout = CLIOption(
        '--dropout', '-d',
        type=float,
        default=0.,
        help='Dropout rate for training.',
    )
    ensemble_size = CLIOption(
        '--ensemble-size', '-z',
        type=int,
        default=1,
        help='Number of models to train in an ensemble.',
    )
    batch_size = CLIOption(
        '--batch', '-b',
        type=int,
        default=16,
        help='Batch size for training.',
    )
    n_epochs = CLIOption(
        '--epochs', '-e',
        type=int,
        default=1,
        help='Number of epochs for training.',
    )
    learning_rate = CLIOption(
        '--learning-rate', '-r',
        type=float,
        default=None,
        help=f'Learning rate for training. Default: {_LR_DEFAULT}.',
    )
    early_stopping = CLIOption(
        '--early-stopping', '-s',
        type=int,
        default=None,
        help='Number of epochs to wait for improvement before stopping. Default: no early stopping.',
    )
    model_config = CLIOption(
        '--config', '-c',
        type=str,
        default=None,
        help='Model configuration file. Overrides other options.',
    )
    config_i = CLIOption(
        '--config-index', '-i',
        type=int,
        default=0,
        help='If more than one config in `--config`, choose this one.',
    )
    output_name = CLIOption(
        '--output', '-o', 
        type=str,
        required=True,
        help='Output filename.',
    )
    serialize = CLIOption(
        '--serialize', '-z', 
        action="store_true",
        help='Pickle instead of JSON output.',
    )
    # output = CLIOption(
    #     '--output', '-o', 
    #     type=FileType('w'),
    #     default=sys.stdout,
    #     help='Output file. Default: STDOUT',
    # )
    # formatting = CLIOption(
    #     '--format', '-f', 
    #     type=str,
    #     default='TSV',
    #     choices=['TSV', 'CSV', 'tsv', 'csv'],
    #     help='Format of files. Default: %(default)s',
    # )

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
        help='Extra columns to retain in prediction table; useful for IDs.',
    )

    # Information metrics
    variance = CLIOption(
        '--variance', 
        action="store_true",
        help='Calculate ensemble variance.',
    )
    tanimoto = CLIOption(
        '--tanimoto', 
        action="store_true",
        help='Calculate Tanimoto distance to nearest neighbor in training data.',
    )
    doubtscore = CLIOption(
        '--doubtscore', 
        action="store_true",
        help='Calculate doubtscore.',
    )
    info_sens = CLIOption(
        '--information-sensitivity', 
        action="store_true",
        help='Calculate information senstivity.',
    )
    optimality = CLIOption(
        '--optimality', 
        action="store_true",
        help='For information sensitivity, make the computationally faster assumption that the model parameters were trained to gradient 0.',
    )
    last_layer = CLIOption(
        '--last-layer',
        action="store_true",
        help="Use only the gradients of parameters in the final layer for doubtscore and info. sens. calculations.",
    )
    hess_approx = CLIOption(
        '--approx', 
        type=str,
        default="bekas",
        choices=["exact_diagonal", "squared_jacobian", "rough_finite_difference", "bekas"],
        help='What type of Hessian approximation to perform for information sensitivity.',
    )
    bekas_n = CLIOption(
        '--bekas-n', 
        type=int,
        default=1,
        help='Number of stochastic samples for Hessian approximation.',
    )

    split_type = CLIOption(
        '--type', 
        type=str,
        default="scaffold",
        choices=["scaffold", "faiss"],
        help='Splitting method.',
    )
    random_seed = CLIOption(
        '--seed', '-i', 
        type=int,
        default=None,
        help='Random seed. Default: determininstic.',
    )
    n_neighbors = CLIOption(
        '--n-neighbors', '-k', 
        type=int,
        default=10,
        help='Number of nearest neighbors for FAISS splitting.',
    )
    train_test_val = [
        CLIOption(
            f'--{key}', 
            type=float,
            default=None,
            help='Fraction of examples for each split. Default: infer.',
        ) for key in ("train", "validation", "test")
    ]

    columns = CLIOption(
        '--columns', '-c',
        type=str,
        nargs='*',
        help='List of columns to tag percentiles for.',
    )
    percentiles = CLIOption(
        '--percentiles', '-p', 
        type=float,
        nargs='*',
        default=[5.],
        help='List of percentiles to calculate.',
    )
    reverse = CLIOption(
        '--reverse', '-r', 
        action='store_true',
        help='Whether to reverse percentiles (i.e. high to low).',
    )
    do_plot = CLIOption(
        '--plot', 
        type=str,
        default=None,
        help='Filename to save UMAP plot under.',
    )
    compression = CLIOption(
        '--compression', '-z', 
        type=int,
        default=500,
        help='How many centroids for quantile approximation. Higher is more accurate, but uses more memory.',
    )
    delta = CLIOption(
        '--delta', '-d', 
        type=float,
        default=1.,
        help='Width from percentile cutoff to buffer as borderline for refinement. Higher is more accurate, but uses more memory.',
    )
    plot_seed = CLIOption(
        '--plot-seed', '-e', 
        type=int,
        default=42,
        help='Seed for UMAP embedding.',
    )
    plot_sample = CLIOption(
        '--plot-sample', '-n', 
        type=int,
        default=20_000,
        help='Subsample size for UMAP embedding.',
    )
    extras = CLIOption(
        '--extras', '-x',
        type=str,
        nargs='*',
        help='Additional columns for coloring UMAP plot.',
    )

    hyperprep = CLICommand(
        "hyperprep",
        description="Prepare inputs for hyperparameter search.",
        options=[
            input_file, 
            serialize,
            output_name,
        ],
        main=_hyperprep,
    )

    train = CLICommand(
        "train",
        description="Train a PyTorch model.",
        options=[
            train_data, 
            val_data,
            test_data,
            feature_cols,
            feature_cols2,
            context,
            structure_col,
            structure_representation,
            label_cols,
            model_class,
            fusion_method,
            _checkpoint,
            n_units,
            n_hidden,
            residual_depth,
            _2d, 
            _fp,
            dropout,
            batch_size,
            n_epochs,
            learning_rate,
            early_stopping,
            ensemble_size,
            model_config,
            config_i,
            output_name,
            cache,
        ],
        main=_train,
    )

    predict = CLICommand(
        "predict",
        description="Make predictions and calculate uncertainty using a duvida checkpoint.",
        options=[
            test_data, 
            slice_start,
            slice_end,
            feature_cols,
            feature_cols2,
            context,
            label_cols,
            structure_col,
            extra_cols,
            structure_representation,
            _checkpoint,
            cache,
            output_name,
            variance,
            tanimoto,
            doubtscore,
            info_sens,
            optimality,
            last_layer,
            hess_approx,
            bekas_n,
            batch_size,
        ],
        main=_predict,
    )

    split = CLICommand(
        "split",
        description="Make chemical train-test-val splits on out-of-core datasets.",
        options=[
            input_filename, 
            split_type,
            n_neighbors,
            slice_start,
            slice_end,
            structure_col,
            structure_representation,
            do_plot,
            plot_sample,
            plot_seed,
            extras,
            random_seed,
            cache,
            output_name,
            batch_size,
        ] + train_test_val,
        main=_split,
    )

    percentiles = CLICommand(
        "percentiles",
        description="Add columns indicating whether rows are in a percentile.",
        options=[
            input_filename, 
            columns,
            percentiles,
            reverse,
            compression,
            delta,
            slice_start,
            slice_end,
            cache,
            output_name,
            batch_size,
            do_plot,
            structure_col,
            structure_representation,
            plot_sample,
            plot_seed,
            extras,
        ],
        main=_percentile,
    )

    app = CLIApp(
        app_name, 
        version=__version__,
        description=(
            "Calculating exact and approximate confidence and "
            "information metrics for deep learning on general "
            "purpose and chemistry tasks."
        ),
        commands=[
            hyperprep,
            train,
            predict,
            split,
            percentiles,
        ],
    )

    app.run()
    return None


if __name__ == '__main__':
    main()
