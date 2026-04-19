from typing import Mapping, Optional, Union

from carabiner import print_err


def _resolve_and_slice_data(
    data: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    batch_size: int = 1024
):
    from ..base.data import DataMixinBase
    # from .utils.datasets import to_dataset

    candidates_ds = DataMixinBase._resolve_data(data)
    nrows = candidates_ds.num_rows
    skip = start or 0
    take = (end or nrows) - skip
    if (take - skip) < nrows:
        print_err(f"INFO: Reading dataset from row {skip} to row {take + skip} / {nrows}.")
    return candidates_ds.skip(skip).take(take)


def _save_dataset(
    dataset,
    output: str
) -> None:
    print_err("INFO: Saving dataset:\n" + str(dataset) + "\n" + f"at {output} as", end=" ")
    if output.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        print_err("CSV.")
        dataset.to_csv(
            output, 
            sep="," if output.endswith((".csv", ".csv.gz")) else "\t",
            compression='gzip' if output.endswith(".gz") else None,
        )
    elif output.endswith(".json"):
        print_err("JSON.")
        dataset.to_json(output)
    elif output.endswith(".parquet"):
        print_err("Parquet.")
        dataset.to_parquet(output)
    elif output.endswith(".sql"):
        print_err("SQL.")
        dataset.to_sql(output)
    elif output.endswith(".hf"):
        print_err("Hugging Face dataset.")
        dataset.save_to_disk(output)
    else:
        print_err("Hugging Face dataset.")
        dataset.save_to_disk(output + ".hf")
        print_err(f"WARNING: Unsure what format to save as for filename {output}. Defaulted to Hugging Face dataset.")
    return None
