from typing import Optional, Iterable, Union

from argparse import Namespace
import os

from carabiner import print_err
from carabiner.cliutils import clicommand

from .io import _resolve_and_slice_data, _save_dataset


def _parse_feature_spec(x: str, i: int):
    if ":" in x:
        input_col, _, remainder = x.partition(":") 
        transforms, _, output_col = remainder.rpartition("@")
    else:
        input_col, _, output_col = x.partition("@")
        transforms, remainder = "", ""
    # print(x, " | ".join([input_col, remainder, transforms, output_col]))
    if not output_col:
        if not transforms:
            output_col = input_col
            extra = True
        else:
            output_col = f"col_{i:03d}"
    else:
        extra = False
    transforms = transforms.split(":")
    if transforms == [""]:
        return extra, {output_col: (input_col, {"name": "identity"})}
    parsed_transforms = []
    for transform in transforms:
        name, _, kwargs = transform.partition("(")
        if not kwargs:
            parsed_transforms.append({"name": name})
            continue
        kwargs = kwargs.removesuffix(")").split(",")
        key_vals = [k.strip().split("=") for k in kwargs]
        wrong_number = [k for k in key_vals if len(k) != 2]
        if wrong_number:
            raise ValueError(f"Badly formatted kwargs: {kwargs=}, parsed to {key_vals}")
        kwargs = {
            key.strip(): value.strip() 
            for key, value in key_vals
        }
        parsed_kwargs = {}
        for k, v in kwargs.items():
            if v.isdigit():
                parsed_v = int(v)
            elif v.casefold() == "true":
                parsed_v = True
            elif v.casefold() == "false":
                parsed_v = False
            else:
                try: 
                    parsed_v = float(v)
                except:
                    parsed_v = v
            parsed_kwargs[k] = parsed_v
        parsed_transforms.append({
            "name": name,
            "kwargs": parsed_kwargs
        })
    return extra, {output_col: (input_col, tuple(parsed_transforms))}


def parse_feature_specs(x: Union[str, Iterable[str]]):
    if isinstance(x, str):
        x = [x]
    column_transforms = []
    extra_cols = []
    for i, _x in enumerate(x):
        extra, column_transform = _parse_feature_spec(_x, i)
        if extra:
            extra_cols.append(_x)
        else:
            column_transforms.append(column_transform)
    return extra_cols, {k: v for d in column_transforms for k, v in d.items()}


def _common_feature_spec_routine(x: Union[str, Iterable[str]], args_extras=None):
    extras, column_transforms = parse_feature_specs(x)
    extras += (args_extras or [])
    extras = sorted(set(extras))

    print_err(
        f"""
        Parsed the feature spec:
            - Column transforms: {column_transforms}
            - Other columns to retain: {extras}

        """
    )
    return extras, column_transforms


def _validate_checkpoints(chk1, path):
    from ..data import DataPipeline
    print_err(f"[INFO] Validating checkpoint at {path}...", end="")
    pipeline2 = DataPipeline().load_checkpoint(path)
    if chk1 != pipeline2:
        attr1, attr2 = vars(chk1), vars(pipeline2)
        wrong_attributes = {
            k: (attr1[k], "!=", v) 
            for k, v in attr2.items() 
            if v != attr1[k]
        }
        raise IOError(
            f"Checkpoint at {path} does not recreate the same object. "
            f"These attributes were different: {wrong_attributes}"
        )
    print_err(" all good!")
    return None


@clicommand("Serializing featurization with the following parameters")
def _serialize(args: Namespace) -> None:

    from ..data import DataPipeline

    output = args.output
    out_dir = os.path.dirname(output)
    base = os.path.basename(output)
    if len(out_dir) > 0:
        os.makedirs(out_dir, exist_ok=True)
    
    extras, column_transforms = _common_feature_spec_routine(
        args.features, 
        args.extras,
    )
    pipeline = DataPipeline(
        column_transforms=column_transforms,
        columns_to_keep=extras,
    )
    pipeline.save_checkpoint(output)
    _validate_checkpoints(pipeline, output)
    return None


@clicommand("Featurizing data with the following parameters")
def _featurize(args: Namespace) -> None:

    from ..data import DataPipeline
    
    if args.features:
        extras, column_transforms = _common_feature_spec_routine(args.features, args.extras)
        pipeline = DataPipeline(
            column_transforms=column_transforms,
            columns_to_keep=extras,
            cache_dir=args.cache,
        )
    elif args.config:
        extras = args.extras or []
        pipeline = DataPipeline(
            cache_dir=args.cache,
        ).load_checkpoint(args.config)
    else:
        raise ValueError(f"One of --features or --config must be provided.")
    
    ds = _resolve_and_slice_data(
        args.input_file,
        start=args.start,
        end=args.end,
        cache_dir=args.cache,
    )
    ds = pipeline(
        ds, 
        keep_extra_columns=extras,
        drop_unused_columns=True,
    )
    if args.checkpoint:
        output = args.checkpoint
        out_dir = os.path.dirname(output)
        base = os.path.basename(output)
        pipeline.save_checkpoint(output)
        _validate_checkpoints(pipeline, output)

    if args.output:
        output = args.output
        out_dir = os.path.dirname(output)
        base = os.path.basename(output)
        if len(out_dir) > 0:
            os.makedirs(out_dir, exist_ok=True)
        _save_dataset(
            ds, 
            output,
        )
    return None
