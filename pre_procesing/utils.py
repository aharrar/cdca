import numpy as np
from sklearn.impute import SimpleImputer


def pre_process_cytof_data(data):
    return np.log2(1 + data)


def standardize_features(tensor):
    """Standardizes all the features in .

  Args:
    tensor: The 2D tensor to standardize.

  Returns:
    The standardized 2D tensor.
  """

    # Get the mean and standard deviation of each column.
    means = tensor.mean(dim=0)
    stds = tensor.std(dim=0)

    # Subtract the mean from each column.
    standardized_tensor = tensor - means

    # Divide each column by the standard deviation.
    standardized_tensor = standardized_tensor / stds

    return standardized_tensor


import torch


def normalize_columns(tensor):
    """Normalizes the columns of a tensor.

  Args:
    tensor: The tensor to normalize.

  Returns:
    The normalized tensor.
  """

    # Get the mean and standard deviation of each column.
    mean = torch.mean(tensor, dim=0, keepdim=True)
    std = torch.std(tensor, dim=0, keepdim=True)

    # Normalize each column.
    normalized_tensor = (tensor - mean) / std

    return normalized_tensor


def remove_nan(data):
    if np.any(np.isnan(data)):
        my_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        data = my_imputer.fit_transform(data)

    return data


def load_and_pre_process_data(src_path, target_path):
    src_data = np.loadtxt(src_path, delimiter=',')
    target_data = np.loadtxt(target_path, delimiter=',')

    src_data = remove_nan(src_data)
    target_data = remove_nan(target_data)

    src_data = pre_process_cytof_data(src_data)
    target_data = pre_process_cytof_data(target_data)

    return src_data, target_data
