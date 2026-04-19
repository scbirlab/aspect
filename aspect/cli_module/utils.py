from typing import Mapping, Optional, Union

from carabiner import pprint_dict

def _overwrite_config(
    config: Mapping, 
    config_file: Optional[str] = None, 
    config_idx: int = 0
) -> dict:

    from ..hyperparameters import HyperOpt

    if config_file is not None:
        new_config = HyperOpt.from_file(config_file, silent=True)._ranges[config_idx] 
        pprint_dict(
            new_config,
            message=f"Overriding command-line parameters from config file {config_file}",
        )
        # new_config.update({key: val in config.items() if val is not None})  # command line takes precedent
        config.update(new_config)  # config takes precedent
        pprint_dict(
            config,
            message="Initialization parameters are now",
        )
        return config
    else:
        return config


def _init_modelbox(
    cli_config: Mapping[str, Union[str, int, float]],
    checkpoint: Optional[str] = None,
    config_file: Optional[str] = None,
    config_idx: int = 0,
    cache: Optional[str] = None,
    **overrides
):
    from ..autoclass import AutoModelBox
    if checkpoint is None:
        if any(
            overrides.get(key) is None for key in ("training", "labels")
        ) and all(
            overrides.get(key) is None
            for key in ("structure", "features")
        ):
                raise ValueError(
                    """If not providing a checkpoint, --training and --labels, 
                    and either --features or --structure must be set.
                    """
                )
        cli_config = _overwrite_config(
            cli_config, 
            config_file=config_file, 
            config_idx=config_idx,
        )
        modelbox = AutoModelBox(**cli_config)._instance
    else:
        modelbox = AutoModelBox.from_pretrained(checkpoint, cache=cache)
    return modelbox


def _dict_to_pandas(
    d: Mapping,
    filename: Optional[str] = None
):
    import pandas as pd
    try:
        df = pd.DataFrame(d)
    except ValueError as e:  # not all columns same length; should never happen
        pprint_dict(
            {key: len(val) for key, val in d.items()},
            message="Metrics table column lengths"
        )
        raise e
    if filename is not None:
        df.to_csv(filename, index=False)
    return df
