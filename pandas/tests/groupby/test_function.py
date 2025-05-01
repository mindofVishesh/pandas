import pandas as pd
import numpy as np
import pytest

def test_groupby_corr_with_timedelta():
    df = pd.DataFrame({
        "group": ["a", "a", "b", "b"],
        "td1": pd.to_timedelta([1, 2, 3, 4], unit="D"),
        "td2": pd.to_timedelta([5, 6, 7, 6], unit="D"),
    })

    result = df.groupby("group")[["td1", "td2"]].corr()

    # Validate that result is not all NaN and contains expected correlation output
    assert isinstance(result, pd.DataFrame)
    assert not result.isnull().all().all(), "Correlation result should not be all NaN"

    # Optional: check values for expected correlation
    a_corr = result.loc["a"]
    b_corr = result.loc["b"]

    assert np.isclose(a_corr.loc["td1", "td2"], 1.0), "Expected perfect correlation in group a"
    assert np.isclose(b_corr.loc["td1", "td2"], -1.0), "Expected negative correlation in group b"
