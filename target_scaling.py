from sklearn.base import clone
from string import digits


def check_scalers(pipeline):
    """Returns list of scalers used in the pipeline in order of appearance."""

    scalers = ["maxabsscaler", "standardscaler", "robustscaler", "minmaxscaler"]
    remove_digits_and_hyphen = str.maketrans('', '', digits + "-")
    list_of_scalers = []
    for step in pipeline.steps:
        # remove numbering in case of duplicated operators
        scaler_name = step[0].translate(remove_digits_and_hyphen)
        if scaler_name in scalers:
            list_of_scalers.append(clone(step[1]))

    return list_of_scalers if len(list_of_scalers) >= 1 else None