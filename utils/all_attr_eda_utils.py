# global imports
# import missingno as msno
# import matplotlib.pyplot as plt
# import numpy as np

# local imports
# import src.utils.lin_reg_diag_utils.lin_reg_diag_utils as lin_reg_diag_utils
# import src.utils.eda_utils.attr_eda_utils as attr_eda_utils


def check_for_complete_unique_attrs(cap_x_df):

    print(f'the data frame has {cap_x_df.shape[0]} rows\n')

    concern_list = []
    for attr in cap_x_df.columns:
        label = ''
        if cap_x_df[attr].nunique() == cap_x_df.shape[0]:
            label = 'examine more closely'
            concern_list.append(attr)
        print(f'{attr} has {cap_x_df[attr].nunique()} unique values and is dtype {cap_x_df[attr].dtype} {label}')

    return concern_list


if __name__ == '__main__':
    pass
