from os import PathLike

import pandas as pd
import numpy as np


# a collection of simple utilities that I didn't want cluttering up the jupyter notebooks.


def normalize_string(s):
    if s.dtypes == 'str': # we might pass columns through that aren't strings
        return s.str.lower() \
            .str.encode('ascii', 'ignore') \
            .str.decode(encoding='ascii') \
            .str.translate(str.maketrans({'"': ''}))
    return s


def load_enrollment(path: str | PathLike = 'enrollments.csv', normalize: bool = True) -> pd.DataFrame:
    result = pd.read_csv(path, low_memory=False)
    return result.apply(normalize_string, axis=0) if normalize else result


def load_gdp(path: str | PathLike = 'Countries GDP 1960-2020.csv', normalize: bool = True) -> pd.DataFrame:
    result = pd.read_csv(path, low_memory=False)
    return result.apply(normalize_string, axis=0) if normalize else result


def make_joined_df(enrollment: pd.DataFrame | None = None, gdp: pd.DataFrame | None = None) -> pd.DataFrame:
    if enrollment is None:
        # enforce normalization to ensure join
        enrollment = load_enrollment(normalize=True)

    if gdp is None:
        gdp = load_gdp(normalize=True)  # enforce normalization to ensure join
        
    # gdp has years as columns, melt them into a long format to align to the enrollment data
    gdp = melt_gdp(gdp)

    # This is a simple join, ignoring the
    return enrollment.merge(gdp, left_on=['country', 'countrycode', 'year'], right_on=['Country Name', 'Country Code', 'year'])


def melt_gdp(x: pd.DataFrame) -> pd.DataFrame:
    tmp = x.melt(id_vars=['Country Name', 'Country Code'], var_name='year', value_name='gdp')
    tmp['year'] = tmp['year'].astype(np.float64) # pd detected mixed types on import, fix it now so it joins correctly
    return tmp


def q1(x): return x.quantile(0.25)  # for inline aggregation


def q3(x): return x.quantile(0.75)  # for inline aggregation
