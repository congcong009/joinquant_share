#
# Copyright 2018 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import re
import warnings

from IPython.display import display
from pandas.tseries.offsets import CustomBusinessDay, Day, BusinessDay
from scipy.stats import mode


class NonMatchingTimezoneError(Exception):
    pass


class MaxLossExceededError(Exception):
    pass


def rethrow(exception, additional_message):
    """
    Re-raise the last exception that was active in the current scope
    without losing the stacktrace but adding an additional message.
    This is hacky because it has to be compatible with both python 2/3
    """
    e = exception
    m = additional_message
    if not e.args:
        e.args = (m,)
    else:
        e.args = (e.args[0] + m,) + e.args[1:]
    raise e


def non_unique_bin_edges_error(func):
    """
    Give user a more informative error in case it is not possible
    to properly calculate quantiles on the input dataframe (factor)
    """
    message = """

    An error occurred while computing bins/quantiles on the input provided.
    This usually happens when the input contains too many identical
    values and they span more than one quantile. The quantiles are choosen
    to have the same number of records each, but the same value cannot span
    multiple quantiles. Possible workarounds are:
    1 - Decrease the number of quantiles
    2 - Specify a custom quantiles range, e.g. [0, .50, .75, 1.] to get unequal
        number of records per quantile
    3 - Use 'bins' option instead of 'quantiles', 'bins' chooses the
        buckets to be evenly spaced according to the values themselves, while
        'quantiles' forces the buckets to have the same number of records.
    4 - for factors with discrete values use the 'bins' option with custom
        ranges and create a range for each discrete value
    Please see utils.get_clean_factor_and_forward_returns documentation for
    full documentation of 'bins' and 'quantiles' options.

"""

    def dec(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if 'Bin edges must be unique' in str(e):
                rethrow(e, message)
            raise
    return dec


@non_unique_bin_edges_error
def quantize_factor(factor_data,
                    quantiles=5,
                    bins=None,
                    by_group=False,
                    no_raise=False,
                    zero_aware=False):
    """
    Computes period wise factor quantiles.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.

        - See full explanation in utils.get_clean_factor_and_forward_returns

    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Only one of 'quantiles' or 'bins' can be not-None
    by_group : bool, optional
        If True, compute quantile buckets separately for each group.
    no_raise: bool, optional
        If True, no exceptions are thrown and the values for which the
        exception would have been thrown are set to np.NaN
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.

    Returns
    -------
    factor_quantile : pd.Series
        Factor quantiles indexed by date and asset.
    """
    if not ((quantiles is not None and bins is None) or
            (quantiles is None and bins is not None)):
        raise ValueError('Either quantiles or bins should be provided')

    if zero_aware and not (isinstance(quantiles, int)
                           or isinstance(bins, int)):
        msg = ("zero_aware should only be True when quantiles or bins is an"
               " integer")
        raise ValueError(msg)

    def quantile_calc(x, _quantiles, _bins, _zero_aware, _no_raise):
        try:
            if _quantiles is not None and _bins is None and not _zero_aware:
                return pd.qcut(x, _quantiles, labels=False) + 1
            elif _quantiles is not None and _bins is None and _zero_aware:
                pos_quantiles = pd.qcut(x[x >= 0], _quantiles // 2,
                                        labels=False) + _quantiles // 2 + 1
                neg_quantiles = pd.qcut(x[x < 0], _quantiles // 2,
                                        labels=False) + 1
                return pd.concat([pos_quantiles, neg_quantiles]).sort_index()
            elif _bins is not None and _quantiles is None and not _zero_aware:
                return pd.cut(x, _bins, labels=False) + 1
            elif _bins is not None and _quantiles is None and _zero_aware:
                pos_bins = pd.cut(x[x >= 0], _bins // 2,
                                  labels=False) + _bins // 2 + 1
                neg_bins = pd.cut(x[x < 0], _bins // 2,
                                  labels=False) + 1
                return pd.concat([pos_bins, neg_bins]).sort_index()
        except Exception as e:
            if _no_raise:
                return pd.Series(index=x.index)
            raise e

    grouper = [factor_data.index.get_level_values('date')]
    if by_group:
        grouper.append('group')

    factor_quantile = factor_data.groupby(grouper)['factor'] \
        .apply(quantile_calc, quantiles, bins, zero_aware, no_raise)
    factor_quantile.name = 'factor_quantile'

    return factor_quantile.dropna()


def infer_trading_calendar(factor_idx, prices_idx):
    """
    Infer the trading calendar from factor and price information.

    Parameters
    ----------
    factor_idx : pd.DatetimeIndex
        The factor datetimes for which we are computing the forward returns
    prices_idx : pd.DatetimeIndex
        The prices datetimes associated withthe factor data

    Returns
    -------
    calendar : pd.DateOffset
    """
    full_idx = factor_idx.union(prices_idx)

    traded_weekdays = []
    holidays = []

    days_of_the_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for day, day_str in enumerate(days_of_the_week):

        weekday_mask = (full_idx.dayofweek == day)

        # drop days of the week that are not traded at all
        if not weekday_mask.any():
            continue
        traded_weekdays.append(day_str)

        # look for holidays
        used_weekdays = full_idx[weekday_mask].normalize()
        all_weekdays = pd.date_range(full_idx.min(), full_idx.max(),
                                     freq=CustomBusinessDay(weekmask=day_str)
                                     ).normalize()
        _holidays = all_weekdays.difference(used_weekdays)
        _holidays = [timestamp.date() for timestamp in _holidays]
        holidays.extend(_holidays)

    traded_weekdays = ' '.join(traded_weekdays)
    return CustomBusinessDay(weekmask=traded_weekdays, holidays=holidays)


def compute_forward_returns(factor,
                            prices,
                            periods=(1, 5, 10),
                            filter_zscore=None,
                            cumulative_returns=True):
    """
    Finds the N period forward returns (as percent change) for each asset
    provided.

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.

        - See full explanation in utils.get_clean_factor_and_forward_returns

    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    cumulative_returns : bool, optional
        If True, forward returns columns will contain cumulative returns.
        Setting this to False is useful if you want to analyze how predictive
        a factor is for a single forward day.

    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by timestamp (level 0) and asset
        (level 1), containing the forward returns for assets.
        Forward returns column names follow the format accepted by
        pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc).
        'date' index freq property (forward_returns.index.levels[0].freq)
        will be set to a trading calendar (pandas DateOffset) inferred
        from the input data (see infer_trading_calendar for more details).
    """

    factor_dateindex = factor.index.levels[0]
    if factor_dateindex.tz != prices.index.tz:
        raise NonMatchingTimezoneError("The timezone of 'factor' is not the "
                                       "same as the timezone of 'prices'. See "
                                       "the pandas methods tz_localize and "
                                       "tz_convert.")

    freq = infer_trading_calendar(factor_dateindex, prices.index)

    factor_dateindex = factor_dateindex.intersection(prices.index)

    if len(factor_dateindex) == 0:
        raise ValueError("Factor and prices indices don't match: make sure "
                         "they have the same convention in terms of datetimes "
                         "and symbol-names")

    # chop prices down to only the assets we care about (= unique assets in
    # `factor`).  we could modify `prices` in place, but that might confuse
    # the caller.
    prices = prices.filter(items=factor.index.levels[1])

    raw_values_dict = {}
    column_list = []

    for period in sorted(periods):
        if cumulative_returns:
            returns = prices.pct_change(period)
        else:
            returns = prices.pct_change()

        forward_returns = \
            returns.shift(-period).reindex(factor_dateindex)

        if filter_zscore is not None:
            mask = abs(
                forward_returns - forward_returns.mean()
            ) > (filter_zscore * forward_returns.std())
            forward_returns[mask] = np.nan

        #
        # Find the period length, which will be the column name. We'll test
        # several entries in order to find out the most likely period length
        # (in case the user passed inconsinstent data)
        #
        days_diffs = []
        for i in range(30):
            if i >= len(forward_returns.index):
                break
            p_idx = prices.index.get_loc(forward_returns.index[i])
            if p_idx is None or p_idx < 0 or (
                    p_idx + period) >= len(prices.index):
                continue
            start = prices.index[p_idx]
            end = prices.index[p_idx + period]
            period_len = diff_custom_calendar_timedeltas(start, end, freq)
            days_diffs.append(period_len.components.days)

        delta_days = period_len.components.days - mode(days_diffs).mode[0]
        period_len -= pd.Timedelta(days=delta_days)
        label = timedelta_to_string(period_len)

        column_list.append(label)

        raw_values_dict[label] = np.concatenate(forward_returns.values)

    df = pd.DataFrame.from_dict(raw_values_dict)
    df.set_index(
        pd.MultiIndex.from_product(
            [factor_dateindex, prices.columns],
            names=['date', 'asset']
        ),
        inplace=True
    )
    df = df.reindex(factor.index)

    # now set the columns correctly
    df = df[column_list]

    df.index.levels[0].freq = freq
    df.index.set_names(['date', 'asset'], inplace=True)

    return df


def backshift_returns_series(series, N):
    """Shift a multi-indexed series backwards by N observations in
    the first level.

    This can be used to convert backward-looking returns into a
    forward-returns series.
    """
    ix = series.index
    dates, sids = ix.levels
    date_labels, sid_labels = map(np.array, ix.labels)

    # Output date labels will contain the all but the last N dates.
    new_dates = dates[:-N]

    # Output data will remove the first M rows, where M is the index of the
    # last record with one of the first N dates.
    cutoff = date_labels.searchsorted(N)
    new_date_labels = date_labels[cutoff:] - N
    new_sid_labels = sid_labels[cutoff:]
    new_values = series.values[cutoff:]

    assert new_date_labels[0] == 0

    new_index = pd.MultiIndex(
        levels=[new_dates, sids],
        labels=[new_date_labels, new_sid_labels],
        sortorder=1,
        names=ix.names,
    )

    return pd.Series(data=new_values, index=new_index)


def demean_forward_returns(factor_data, grouper=None):
    """
    Convert forward returns to returns relative to mean
    period wise all-universe or group returns.
    group-wise normalization incorporates the assumption of a
    group neutral portfolio constraint and thus allows allows the
    factor to be evaluated across groups.

    For example, if AAPL 5 period return is 0.1% and mean 5 period
    return for the Technology stocks in our universe was 0.5% in the
    same period, the group adjusted 5 period return for AAPL in this
    period is -0.4%.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        Forward returns indexed by date and asset.
        Separate column for each forward return window.
    grouper : list
        If True, demean according to group.

    Returns
    -------
    adjusted_forward_returns : pd.DataFrame - MultiIndex
        DataFrame of the same format as the input, but with each
        security's returns normalized by group.
    """

    factor_data = factor_data.copy()

    if not grouper:
        grouper = factor_data.index.get_level_values('date')

    cols = get_forward_returns_columns(factor_data.columns)
    factor_data[cols] = factor_data.groupby(grouper)[cols] \
        .transform(lambda x: x - x.mean())

    return factor_data


def print_table(table, name=None, fmt=None):
    """
    Pretty print a pandas DataFrame.

    Uses HTML output if running inside Jupyter Notebook, otherwise
    formatted text output.

    Parameters
    ----------
    table : pd.Series or pd.DataFrame
        Table to pretty-print.
    name : str, optional
        Table name to display in upper left corner.
    fmt : str, optional
        Formatter to use for displaying table elements.
        E.g. '{0:.2f}%' for displaying 100 as '100.00%'.
        Restores original setting after displaying.
    """
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if isinstance(table, pd.DataFrame):
        table.columns.name = name

    prev_option = pd.get_option('display.float_format')
    if fmt is not None:
        pd.set_option('display.float_format', lambda x: fmt.format(x))

    display(table)

    if fmt is not None:
        pd.set_option('display.float_format', prev_option)


def get_clean_factor(factor,
                     forward_returns,
                     groupby=None,
                     binning_by_group=False,
                     quantiles=5,
                     bins=None,
                     groupby_labels=None,
                     max_loss=0.35,
                     zero_aware=False):
    """
    Formats the factor data, forward return data, and group mappings into a
    DataFrame that contains aligned MultiIndex indices of timestamp and asset.
    The returned data will be formatted to be suitable for Alphalens functions.

    It is safe to skip a call to this function and still make use of Alphalens
    functionalities as long as the factor data conforms to the format returned
    from get_clean_factor_and_forward_returns and documented here

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------

    forward_returns : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by timestamp (level 0) and asset
        (level 1), containing the forward returns for assets.
        Forward returns column names must follow the format accepted by
        pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc).
        'date' index freq property must be set to a trading calendar
        (pandas DateOffset), see infer_trading_calendar for more details.
        This information is currently used only in cumulative returns
        computation
        ::
            ---------------------------------------
                       |       | 1D  | 5D  | 10D
            ---------------------------------------
                date   | asset |     |     |
            ---------------------------------------
                       | AAPL  | 0.09|-0.01|-0.079
                       ----------------------------
                       | BA    | 0.02| 0.06| 0.020
                       ----------------------------
            2014-01-01 | CMG   | 0.03| 0.09| 0.036
                       ----------------------------
                       | DAL   |-0.02|-0.06|-0.029
                       ----------------------------
                       | LULU  |-0.03| 0.05|-0.009
                       ----------------------------

    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    binning_by_group : bool
        If True, compute quantile buckets separately for each group.
        This is useful when the factor values range vary considerably
        across gorups so that it is wise to make the binning group relative.
        You should probably enable this if the factor is intended
        to be analyzed for a group neutral portfolio
    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Chooses the buckets to be evenly spaced according to the values
        themselves. Useful when the factor contains discrete values.
        Only one of 'quantiles' or 'bins' can be not-None
    groupby_labels : dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.
    max_loss : float, optional
        Maximum percentage (0.00 to 1.00) of factor data dropping allowed,
        computed comparing the number of items in the input factor index and
        the number of items in the output DataFrame index.
        Factor data can be partially dropped due to being flawed itself
        (e.g. NaNs), not having provided enough price data to compute
        forward returns for all factor values, or because it is not possible
        to perform binning.
        Set max_loss=0 to avoid Exceptions suppression.
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.
        'quantiles' is None.

    Returns
    -------
    merged_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.

        - forward returns column names follow the format accepted by
          pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc)

        - 'date' index freq property (merged_data.index.levels[0].freq) is the
          same as that of the input forward returns data. This is currently
          used only in cumulative returns computation
        ::
           -------------------------------------------------------------------
                      |       | 1D  | 5D  | 10D  |factor|group|factor_quantile
           -------------------------------------------------------------------
               date   | asset |     |     |      |      |     |
           -------------------------------------------------------------------
                      | AAPL  | 0.09|-0.01|-0.079|  0.5 |  G1 |      3
                      --------------------------------------------------------
                      | BA    | 0.02| 0.06| 0.020| -1.1 |  G2 |      5
                      --------------------------------------------------------
           2014-01-01 | CMG   | 0.03| 0.09| 0.036|  1.7 |  G2 |      1
                      --------------------------------------------------------
                      | DAL   |-0.02|-0.06|-0.029| -0.1 |  G3 |      5
                      --------------------------------------------------------
                      | LULU  |-0.03| 0.05|-0.009|  2.7 |  G1 |      2
                      --------------------------------------------------------
    """

    initial_amount = float(len(factor.index))

    factor_copy = factor.copy()
    factor_copy.index = factor_copy.index.rename(['date', 'asset'])
    factor_copy = factor_copy[np.isfinite(factor_copy)]

    merged_data = forward_returns.copy()
    merged_data['factor'] = factor_copy

    if groupby is not None:
        if isinstance(groupby, dict):
            diff = set(factor_copy.index.get_level_values(
                'asset')) - set(groupby.keys())
            if len(diff) > 0:
                raise KeyError(
                    "Assets {} not in group mapping".format(
                        list(diff)))

            ss = pd.Series(groupby)
            groupby = pd.Series(index=factor_copy.index,
                                data=ss[factor_copy.index.get_level_values(
                                    'asset')].values)

        if groupby_labels is not None:
            diff = set(groupby.values) - set(groupby_labels.keys())
            if len(diff) > 0:
                raise KeyError(
                    "groups {} not in passed group names".format(
                        list(diff)))

            sn = pd.Series(groupby_labels)
            groupby = pd.Series(index=groupby.index,
                                data=sn[groupby.values].values)

        merged_data['group'] = groupby.astype('category')

    merged_data = merged_data.dropna()

    fwdret_amount = float(len(merged_data.index))

    no_raise = False if max_loss == 0 else True
    quantile_data = quantize_factor(
        merged_data,
        quantiles,
        bins,
        binning_by_group,
        no_raise,
        zero_aware
    )

    merged_data['factor_quantile'] = quantile_data

    merged_data = merged_data.dropna()

    binning_amount = float(len(merged_data.index))

    tot_loss = (initial_amount - binning_amount) / initial_amount
    fwdret_loss = (initial_amount - fwdret_amount) / initial_amount
    bin_loss = tot_loss - fwdret_loss

    print("Dropped %.1f%% entries from factor data: %.1f%% in forward "
          "returns computation and %.1f%% in binning phase "
          "(set max_loss=0 to see potentially suppressed Exceptions)." %
          (tot_loss * 100, fwdret_loss * 100, bin_loss * 100))

    if tot_loss > max_loss:
        message = ("max_loss (%.1f%%) exceeded %.1f%%, consider increasing it."
                   % (max_loss * 100, tot_loss * 100))
        raise MaxLossExceededError(message)
    else:
        print("max_loss is %.1f%%, not exceeded: OK!" % (max_loss * 100))

    return merged_data


def get_clean_factor_and_forward_returns(factor,
                                         prices,
                                         groupby=None,
                                         binning_by_group=False,
                                         quantiles=5,
                                         bins=None,
                                         periods=(1, 5, 10),
                                         filter_zscore=20,
                                         groupby_labels=None,
                                         max_loss=0.35,
                                         zero_aware=False,
                                         cumulative_returns=True):
    """
    Formats the factor data, pricing data, and group mappings into a DataFrame
    that contains aligned MultiIndex indices of timestamp and asset. The
    returned data will be formatted to be suitable for Alphalens functions.

    It is safe to skip a call to this function and still make use of Alphalens
    functionalities as long as the factor data conforms to the format returned
    from get_clean_factor_and_forward_returns and documented here

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------

    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by timestamp with assets
        in the columns.
        Pricing data must span the factor analysis time period plus an
        additional buffer window that is greater than the maximum number
        of expected periods in the forward returns calculations.
        It is important to pass the correct pricing data in depending on
        what time of period your signal was generated so to avoid lookahead
        bias, or  delayed calculations.
        'Prices' must contain at least an entry for each timestamp/asset
        combination in 'factor'. This entry should reflect the buy price
        for the assets and usually it is the next available price after the
        factor is computed but it can also be a later price if the factor is
        meant to be traded later (e.g. if the factor is computed at market
        open but traded 1 hour after market open the price information should
        be 1 hour after market open).
        'Prices' must also contain entries for timestamps following each
        timestamp/asset combination in 'factor', as many more timestamps
        as the maximum value in 'periods'. The asset price after 'period'
        timestamps will be considered the sell price for that asset when
        computing 'period' forward returns.
        ::
            ----------------------------------------------------
                        | AAPL |  BA  |  CMG  |  DAL  |  LULU  |
            ----------------------------------------------------
               Date     |      |      |       |       |        |
            ----------------------------------------------------
            2014-01-01  |605.12| 24.58|  11.72| 54.43 |  37.14 |
            ----------------------------------------------------
            2014-01-02  |604.35| 22.23|  12.21| 52.78 |  33.63 |
            ----------------------------------------------------
            2014-01-03  |607.94| 21.68|  14.36| 53.94 |  29.37 |
            ----------------------------------------------------

    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    binning_by_group : bool
        If True, compute quantile buckets separately for each group.
        This is useful when the factor values range vary considerably
        across gorups so that it is wise to make the binning group relative.
        You should probably enable this if the factor is intended
        to be analyzed for a group neutral portfolio
    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Chooses the buckets to be evenly spaced according to the values
        themselves. Useful when the factor contains discrete values.
        Only one of 'quantiles' or 'bins' can be not-None
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    groupby_labels : dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.
    max_loss : float, optional
        Maximum percentage (0.00 to 1.00) of factor data dropping allowed,
        computed comparing the number of items in the input factor index and
        the number of items in the output DataFrame index.
        Factor data can be partially dropped due to being flawed itself
        (e.g. NaNs), not having provided enough price data to compute
        forward returns for all factor values, or because it is not possible
        to perform binning.
        Set max_loss=0 to avoid Exceptions suppression.
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.
    cumulative_returns : bool, optional
        If True, forward returns columns will contain cumulative returns.
        Setting this to False is useful if you want to analyze how predictive
        a factor is for a single forward day.

    Returns
    -------
    merged_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - forward returns column names follow  the format accepted by
          pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc)
        - 'date' index freq property (merged_data.index.levels[0].freq) will be
          set to a trading calendar (pandas DateOffset) inferred from the input
          data (see infer_trading_calendar for more details). This is currently
          used only in cumulative returns computation
        ::
           -------------------------------------------------------------------
                      |       | 1D  | 5D  | 10D  |factor|group|factor_quantile
           -------------------------------------------------------------------
               date   | asset |     |     |      |      |     |
           -------------------------------------------------------------------
                      | AAPL  | 0.09|-0.01|-0.079|  0.5 |  G1 |      3
                      --------------------------------------------------------
                      | BA    | 0.02| 0.06| 0.020| -1.1 |  G2 |      5
                      --------------------------------------------------------
           2014-01-01 | CMG   | 0.03| 0.09| 0.036|  1.7 |  G2 |      1
                      --------------------------------------------------------
                      | DAL   |-0.02|-0.06|-0.029| -0.1 |  G3 |      5
                      --------------------------------------------------------
                      | LULU  |-0.03| 0.05|-0.009|  2.7 |  G1 |      2
                      --------------------------------------------------------

    See Also
    --------
    utils.get_clean_factor
        For use when forward returns are already available.
    """
    forward_returns = compute_forward_returns(
        factor,
        prices,
        periods,
        filter_zscore,
        cumulative_returns,
    )

    factor_data = get_clean_factor(factor, forward_returns, groupby=groupby,
                                   groupby_labels=groupby_labels,
                                   quantiles=quantiles, bins=bins,
                                   binning_by_group=binning_by_group,
                                   max_loss=max_loss, zero_aware=zero_aware)

    return factor_data


def rate_of_return(period_ret, base_period):
    """
    Convert returns to 'one_period_len' rate of returns: that is the value the
    returns would have every 'one_period_len' if they had grown at a steady
    rate

    Parameters
    ----------
    period_ret: pd.DataFrame
        DataFrame containing returns values with column headings representing
        the return period.
    base_period: string
        The base period length used in the conversion
        It must follow pandas.Timedelta constructor format (e.g. '1 days',
        '1D', '30m', '3h', '1D1h', etc)

    Returns
    -------
    pd.DataFrame
        DataFrame in same format as input but with 'one_period_len' rate of
        returns values.
    """
    period_len = period_ret.name
    conversion_factor = (pd.Timedelta(base_period) /
                         pd.Timedelta(period_len))
    return period_ret.add(1).pow(conversion_factor).sub(1)


def std_conversion(period_std, base_period):
    """
    one_period_len standard deviation (or standard error) approximation

    Parameters
    ----------
    period_std: pd.DataFrame
        DataFrame containing standard deviation or standard error values
        with column headings representing the return period.
    base_period: string
        The base period length used in the conversion
        It must follow pandas.Timedelta constructor format (e.g. '1 days',
        '1D', '30m', '3h', '1D1h', etc)

    Returns
    -------
    pd.DataFrame
        DataFrame in same format as input but with one-period
        standard deviation/error values.
    """
    period_len = period_std.name
    conversion_factor = (pd.Timedelta(period_len) /
                         pd.Timedelta(base_period))
    return period_std / np.sqrt(conversion_factor)


def get_forward_returns_columns(columns, require_exact_day_multiple=False):
    """
    Utility that detects and returns the columns that are forward returns
    """

    # If exact day multiples are required in the forward return periods,
    # drop all other columns (e.g. drop 3D12h).
    if require_exact_day_multiple:
        pattern = re.compile(r"^(\d+([D]))+$", re.IGNORECASE)
        valid_columns = [(pattern.match(col) is not None) for col in columns]

        if sum(valid_columns) < len(valid_columns):
            warnings.warn(
                "Skipping return periods that aren't exact multiples"
                + " of days."
            )
    else:
        pattern = re.compile(r"^(\d+([Dhms]|ms|us|ns]))+$", re.IGNORECASE)
        valid_columns = [(pattern.match(col) is not None) for col in columns]

    return columns[valid_columns]


def timedelta_to_string(timedelta):
    """
    Utility that converts a pandas.Timedelta to a string representation
    compatible with pandas.Timedelta constructor format

    Parameters
    ----------
    timedelta: pd.Timedelta

    Returns
    -------
    string
        string representation of 'timedelta'
    """
    c = timedelta.components
    format = ''
    if c.days != 0:
        format += '%dD' % c.days
    if c.hours > 0:
        format += '%dh' % c.hours
    if c.minutes > 0:
        format += '%dm' % c.minutes
    if c.seconds > 0:
        format += '%ds' % c.seconds
    if c.milliseconds > 0:
        format += '%dms' % c.milliseconds
    if c.microseconds > 0:
        format += '%dus' % c.microseconds
    if c.nanoseconds > 0:
        format += '%dns' % c.nanoseconds
    return format


def timedelta_strings_to_integers(sequence):
    """
    Converts pandas string representations of timedeltas into integers of days.

    Parameters
    ----------
    sequence : iterable
        List or array of timedelta string representations, e.g. ['1D', '5D'].

    Returns
    -------
    sequence : list
        Integer days corresponding to the input sequence, e.g. [1, 5].
    """
    return list(map(lambda x: pd.Timedelta(x).days, sequence))


def add_custom_calendar_timedelta(input, timedelta, freq):
    """
    Add timedelta to 'input' taking into consideration custom frequency, which
    is used to deal with custom calendars, such as a trading calendar

    Parameters
    ----------
    input : pd.DatetimeIndex or pd.Timestamp
    timedelta : pd.Timedelta
    freq : pd.DataOffset (CustomBusinessDay, Day or BusinessDay)

    Returns
    -------
    pd.DatetimeIndex or pd.Timestamp
        input + timedelta
    """
    if not isinstance(freq, (Day, BusinessDay, CustomBusinessDay)):
        raise ValueError("freq must be Day, BDay or CustomBusinessDay")
    days = timedelta.components.days
    offset = timedelta - pd.Timedelta(days=days)
    return input + freq * days + offset


def diff_custom_calendar_timedeltas(start, end, freq):
    """
    Compute the difference between two pd.Timedelta taking into consideration
    custom frequency, which is used to deal with custom calendars, such as a
    trading calendar

    Parameters
    ----------
    start : pd.Timestamp
    end : pd.Timestamp
    freq : CustomBusinessDay (see infer_trading_calendar)
    freq : pd.DataOffset (CustomBusinessDay, Day or BDay)

    Returns
    -------
    pd.Timedelta
        end - start
    """
    if not isinstance(freq, (Day, BusinessDay, CustomBusinessDay)):
        raise ValueError("freq must be Day, BusinessDay or CustomBusinessDay")

    weekmask = getattr(freq, 'weekmask', None)
    holidays = getattr(freq, 'holidays', None)

    if weekmask is None and holidays is None:
        if isinstance(freq, Day):
            weekmask = 'Mon Tue Wed Thu Fri Sat Sun'
            holidays = []
        elif isinstance(freq, BusinessDay):
            weekmask = 'Mon Tue Wed Thu Fri'
            holidays = []

    if weekmask is not None and holidays is not None:
        # we prefer this method as it is faster
        actual_days = np.busday_count(np.array(start).astype('datetime64[D]'),
                                      np.array(end).astype('datetime64[D]'),
                                      weekmask, holidays)
    else:
        # default, it is slow
        actual_days = pd.date_range(start, end, freq=freq).shape[0] - 1
        if not freq.onOffset(start):
            actual_days -= 1

    timediff = end - start
    delta_days = timediff.components.days - actual_days
    return timediff - pd.Timedelta(days=delta_days)


# -----------------------------------------------------------------------------------------------------------------
#          performance
# -----------------------------------------------------------------------------------------------------------------

#
# Copyright 2017 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import warnings

import empyrical as ep
from pandas.tseries.offsets import BDay
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from . import utils


def factor_information_coefficient(factor_data,
                                   group_adjust=False,
                                   by_group=False):
    """
    Computes the Spearman Rank Correlation based Information Coefficient (IC)
    between factor values and N period forward returns for each period in
    the factor index.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    group_adjust : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, compute period wise IC separately for each group.

    Returns
    -------
    ic : pd.DataFrame
        Spearman Rank correlation between factor and
        provided forward returns.
    """

    def src_ic(group):
        f = group['factor']
        _ic = group[utils.get_forward_returns_columns(factor_data.columns)] \
            .apply(lambda x: stats.spearmanr(x, f)[0])
        return _ic

    factor_data = factor_data.copy()

    grouper = [factor_data.index.get_level_values('date')]

    if group_adjust:
        factor_data = utils.demean_forward_returns(factor_data,
                                                   grouper + ['group'])
    if by_group:
        grouper.append('group')

    ic = factor_data.groupby(grouper).apply(src_ic)

    return ic


def mean_information_coefficient(factor_data,
                                 group_adjust=False,
                                 by_group=False,
                                 by_time=None):
    """
    Get the mean information coefficient of specified groups.
    Answers questions like:
    What is the mean IC for each month?
    What is the mean IC for each group for our whole timerange?
    What is the mean IC for for each group, each week?

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    group_adjust : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, take the mean IC for each group.
    by_time : str (pd time_rule), optional
        Time window to use when taking mean IC.
        See http://pandas.pydata.org/pandas-docs/stable/timeseries.html
        for available options.

    Returns
    -------
    ic : pd.DataFrame
        Mean Spearman Rank correlation between factor and provided
        forward price movement windows.
    """

    ic = factor_information_coefficient(factor_data, group_adjust, by_group)

    grouper = []
    if by_time is not None:
        grouper.append(pd.Grouper(freq=by_time))
    if by_group:
        grouper.append('group')

    if len(grouper) == 0:
        ic = ic.mean()

    else:
        ic = (ic.reset_index().set_index('date').groupby(grouper).mean())

    return ic


def factor_weights(factor_data,
                   demeaned=True,
                   group_adjust=False,
                   equal_weight=False):
    """
    Computes asset weights by factor values and dividing by the sum of their
    absolute value (achieving gross leverage of 1). Positive factor values will
    results in positive weights and negative values in negative weights.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    demeaned : bool
        Should this computation happen on a long short portfolio? if True,
        weights are computed by demeaning factor values and dividing by the sum
        of their absolute value (achieving gross leverage of 1). The sum of
        positive weights will be the same as the negative weights (absolute
        value), suitable for a dollar neutral long-short portfolio
    group_adjust : bool
        Should this computation happen on a group neutral portfolio? If True,
        compute group neutral weights: each group will weight the same and
        if 'demeaned' is enabled the factor values demeaning will occur on the
        group level.
    equal_weight : bool, optional
        if True the assets will be equal-weighted instead of factor-weighted
        If demeaned is True then the factor universe will be split in two
        equal sized groups, top assets with positive weights and bottom assets
        with negative weights

    Returns
    -------
    returns : pd.Series
        Assets weighted by factor value.
    """

    def to_weights(group, _demeaned, _equal_weight):

        if _equal_weight:
            group = group.copy()

            if _demeaned:
                # top assets positive weights, bottom ones negative
                group = group - group.median()

            negative_mask = group < 0
            group[negative_mask] = -1.0
            positive_mask = group > 0
            group[positive_mask] = 1.0

            if _demeaned:
                # positive weights must equal negative weights
                if negative_mask.any():
                    group[negative_mask] /= negative_mask.sum()
                if positive_mask.any():
                    group[positive_mask] /= positive_mask.sum()

        elif _demeaned:
            group = group - group.mean()

        return group / group.abs().sum()

    grouper = [factor_data.index.get_level_values('date')]
    if group_adjust:
        grouper.append('group')

    weights = factor_data.groupby(grouper)['factor'] \
        .apply(to_weights, demeaned, equal_weight)

    if group_adjust:
        weights = weights.groupby(level='date').apply(to_weights, False, False)

    return weights


def factor_returns(factor_data,
                   demeaned=True,
                   group_adjust=False,
                   equal_weight=False,
                   by_asset=False):
    """
    Computes period wise returns for portfolio weighted by factor
    values.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    demeaned : bool
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    group_adjust : bool
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    equal_weight : bool, optional
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    by_asset: bool, optional
        If True, returns are reported separately for each esset.

    Returns
    -------
    returns : pd.DataFrame
        Period wise factor returns
    """

    weights = \
        factor_weights(factor_data, demeaned, group_adjust, equal_weight)

    weighted_returns = \
        factor_data[utils.get_forward_returns_columns(factor_data.columns)] \
        .multiply(weights, axis=0)

    if by_asset:
        returns = weighted_returns
    else:
        returns = weighted_returns.groupby(level='date').sum()

    return returns


def factor_alpha_beta(factor_data,
                      returns=None,
                      demeaned=True,
                      group_adjust=False,
                      equal_weight=False):
    """
    Compute the alpha (excess returns), alpha t-stat (alpha significance),
    and beta (market exposure) of a factor. A regression is run with
    the period wise factor universe mean return as the independent variable
    and mean period wise return from a portfolio weighted by factor values
    as the dependent variable.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    returns : pd.DataFrame, optional
        Period wise factor returns. If this is None then it will be computed
        with 'factor_returns' function and the passed flags: 'demeaned',
        'group_adjust', 'equal_weight'
    demeaned : bool
        Control how to build factor returns used for alpha/beta computation
        -- see performance.factor_return for a full explanation
    group_adjust : bool
        Control how to build factor returns used for alpha/beta computation
        -- see performance.factor_return for a full explanation
    equal_weight : bool, optional
        Control how to build factor returns used for alpha/beta computation
        -- see performance.factor_return for a full explanation

    Returns
    -------
    alpha_beta : pd.Series
        A list containing the alpha, beta, a t-stat(alpha)
        for the given factor and forward returns.
    """

    if returns is None:
        returns = \
            factor_returns(factor_data, demeaned, group_adjust, equal_weight)

    universe_ret = factor_data.groupby(level='date')[
        utils.get_forward_returns_columns(factor_data.columns)] \
        .mean().loc[returns.index]

    if isinstance(returns, pd.Series):
        returns.name = universe_ret.columns.values[0]
        returns = pd.DataFrame(returns)

    alpha_beta = pd.DataFrame()
    for period in returns.columns.values:
        x = universe_ret[period].values
        y = returns[period].values
        x = add_constant(x)

        reg_fit = OLS(y, x).fit()
        try:
            alpha, beta = reg_fit.params
        except ValueError:
            alpha_beta.loc['Ann. alpha', period] = np.nan
            alpha_beta.loc['beta', period] = np.nan
        else:
            freq_adjust = pd.Timedelta('252Days') / pd.Timedelta(period)

            alpha_beta.loc['Ann. alpha', period] = \
                (1 + alpha) ** freq_adjust - 1
            alpha_beta.loc['beta', period] = beta

    return alpha_beta


def cumulative_returns(returns):
    """
    Computes cumulative returns from simple daily returns.

    Parameters
    ----------
    returns: pd.Series
        pd.Series containing daily factor returns (i.e. '1D' returns).

    Returns
    -------
    Cumulative returns series : pd.Series
        Example:
            2015-01-05   1.001310
            2015-01-06   1.000805
            2015-01-07   1.001092
            2015-01-08   0.999200
    """

    return ep.cum_returns(returns, starting_value=1)


def positions(weights, period, freq=None):
    """
    Builds net position values time series, the portfolio percentage invested
    in each position.

    Parameters
    ----------
    weights: pd.Series
        pd.Series containing factor weights, the index contains timestamps at
        which the trades are computed and the values correspond to assets
        weights
        - see factor_weights for more details
    period: pandas.Timedelta or string
        Assets holding period (1 day, 2 mins, 3 hours etc). It can be a
        Timedelta or a string in the format accepted by Timedelta constructor
        ('1 days', '1D', '30m', '3h', '1D1h', etc)
    freq : pandas DateOffset, optional
        Used to specify a particular trading calendar. If not present
        weights.index.freq will be used

    Returns
    -------
    pd.DataFrame
        Assets positions series, datetime on index, assets on columns.
        Example:
            index                 'AAPL'         'MSFT'          cash
            2004-01-09 10:30:00   13939.3800     -14012.9930     711.5585
            2004-01-09 15:30:00       0.00       -16012.9930     411.5585
            2004-01-12 10:30:00   14492.6300     -14624.8700       0.0
            2004-01-12 15:30:00   14874.5400     -15841.2500       0.0
            2004-01-13 10:30:00   -13853.2800    13653.6400      -43.6375
    """

    weights = weights.unstack()

    if not isinstance(period, pd.Timedelta):
        period = pd.Timedelta(period)

    if freq is None:
        freq = weights.index.freq

    if freq is None:
        freq = BDay()
        warnings.warn("'freq' not set, using business day calendar",
                      UserWarning)

    #
    # weights index contains factor computation timestamps, then add returns
    # timestamps too (factor timestamps + period) and save them to 'full_idx'
    # 'full_idx' index will contain an entry for each point in time the weights
    # change and hence they have to be re-computed
    #
    trades_idx = weights.index.copy()
    returns_idx = utils.add_custom_calendar_timedelta(trades_idx, period, freq)
    weights_idx = trades_idx.union(returns_idx)

    #
    # Compute portfolio weights for each point in time contained in the index
    #
    portfolio_weights = pd.DataFrame(index=weights_idx,
                                     columns=weights.columns)
    active_weights = []

    for curr_time in weights_idx:

        #
        # fetch new weights that become available at curr_time and store them
        # in active weights
        #
        if curr_time in weights.index:
            assets_weights = weights.loc[curr_time]
            expire_ts = utils.add_custom_calendar_timedelta(curr_time,
                                                            period, freq)
            active_weights.append((expire_ts, assets_weights))

        #
        # remove expired entry in active_weights (older than 'period')
        #
        if active_weights:
            expire_ts, assets_weights = active_weights[0]
            if expire_ts <= curr_time:
                active_weights.pop(0)

        if not active_weights:
            continue
        #
        # Compute total weights for curr_time and store them
        #
        tot_weights = [w for (ts, w) in active_weights]
        tot_weights = pd.concat(tot_weights, axis=1)
        tot_weights = tot_weights.sum(axis=1)
        tot_weights /= tot_weights.abs().sum()

        portfolio_weights.loc[curr_time] = tot_weights

    return portfolio_weights.fillna(0)


def mean_return_by_quantile(factor_data,
                            by_date=False,
                            by_group=False,
                            demeaned=True,
                            group_adjust=False):
    """
    Computes mean returns for factor quantiles across
    provided forward returns columns.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    by_date : bool
        If True, compute quantile bucket returns separately for each date.
    by_group : bool
        If True, compute quantile bucket returns separately for each group.
    demeaned : bool
        Compute demeaned mean returns (long short portfolio)
    group_adjust : bool
        Returns demeaning will occur on the group level.

    Returns
    -------
    mean_ret : pd.DataFrame
        Mean period wise returns by specified factor quantile.
    std_error_ret : pd.DataFrame
        Standard error of returns by specified quantile.
    """

    if group_adjust:
        grouper = [factor_data.index.get_level_values('date')] + ['group']
        factor_data = utils.demean_forward_returns(factor_data, grouper)
    elif demeaned:
        factor_data = utils.demean_forward_returns(factor_data)
    else:
        factor_data = factor_data.copy()

    grouper = ['factor_quantile', factor_data.index.get_level_values('date')]

    if by_group:
        grouper.append('group')

    group_stats = factor_data.groupby(grouper)[
        utils.get_forward_returns_columns(factor_data.columns)] \
        .agg(['mean', 'std', 'count'])

    mean_ret = group_stats.T.xs('mean', level=1).T

    if not by_date:
        grouper = [mean_ret.index.get_level_values('factor_quantile')]
        if by_group:
            grouper.append(mean_ret.index.get_level_values('group'))
        group_stats = mean_ret.groupby(grouper)\
            .agg(['mean', 'std', 'count'])
        mean_ret = group_stats.T.xs('mean', level=1).T

    std_error_ret = group_stats.T.xs('std', level=1).T \
        / np.sqrt(group_stats.T.xs('count', level=1).T)

    return mean_ret, std_error_ret


def compute_mean_returns_spread(mean_returns,
                                upper_quant,
                                lower_quant,
                                std_err=None):
    """
    Computes the difference between the mean returns of
    two quantiles. Optionally, computes the standard error
    of this difference.

    Parameters
    ----------
    mean_returns : pd.DataFrame
        DataFrame of mean period wise returns by quantile.
        MultiIndex containing date and quantile.
        See mean_return_by_quantile.
    upper_quant : int
        Quantile of mean return from which we
        wish to subtract lower quantile mean return.
    lower_quant : int
        Quantile of mean return we wish to subtract
        from upper quantile mean return.
    std_err : pd.DataFrame, optional
        Period wise standard error in mean return by quantile.
        Takes the same form as mean_returns.

    Returns
    -------
    mean_return_difference : pd.Series
        Period wise difference in quantile returns.
    joint_std_err : pd.Series
        Period wise standard error of the difference in quantile returns.
        if std_err is None, this will be None
    """

    mean_return_difference = mean_returns.xs(upper_quant,
                                             level='factor_quantile') \
        - mean_returns.xs(lower_quant, level='factor_quantile')

    if std_err is None:
        joint_std_err = None
    else:
        std1 = std_err.xs(upper_quant, level='factor_quantile')
        std2 = std_err.xs(lower_quant, level='factor_quantile')
        joint_std_err = np.sqrt(std1**2 + std2**2)

    return mean_return_difference, joint_std_err


def quantile_turnover(quantile_factor, quantile, period=1):
    """
    Computes the proportion of names in a factor quantile that were
    not in that quantile in the previous period.

    Parameters
    ----------
    quantile_factor : pd.Series
        DataFrame with date, asset and factor quantile.
    quantile : int
        Quantile on which to perform turnover analysis.
    period: int, optional
        Number of days over which to calculate the turnover.

    Returns
    -------
    quant_turnover : pd.Series
        Period by period turnover for that quantile.
    """

    quant_names = quantile_factor[quantile_factor == quantile]
    quant_name_sets = quant_names.groupby(level=['date']).apply(
        lambda x: set(x.index.get_level_values('asset')))

    name_shifted = quant_name_sets.shift(period)

    new_names = (quant_name_sets - name_shifted).dropna()
    quant_turnover = new_names.apply(
        lambda x: len(x)) / quant_name_sets.apply(lambda x: len(x))
    quant_turnover.name = quantile
    return quant_turnover


def factor_rank_autocorrelation(factor_data, period=1):
    """
    Computes autocorrelation of mean factor ranks in specified time spans.
    We must compare period to period factor ranks rather than factor values
    to account for systematic shifts in the factor values of all names or names
    within a group. This metric is useful for measuring the turnover of a
    factor. If the value of a factor for each name changes randomly from period
    to period, we'd expect an autocorrelation of 0.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    period: int, optional
        Number of days over which to calculate the turnover.

    Returns
    -------
    autocorr : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation of
        factor values.
    """
    grouper = [factor_data.index.get_level_values('date')]

    ranks = factor_data.groupby(grouper)['factor'].rank()

    asset_factor_rank = ranks.reset_index().pivot(index='date',
                                                  columns='asset',
                                                  values='factor')

    asset_shifted = asset_factor_rank.shift(period)

    autocorr = asset_factor_rank.corrwith(asset_shifted, axis=1)
    autocorr.name = period
    return autocorr


def common_start_returns(factor,
                         returns,
                         before,
                         after,
                         cumulative=False,
                         mean_by_date=False,
                         demean_by=None):
    """
    A date and equity pair is extracted from each index row in the factor
    dataframe and for each of these pairs a return series is built starting
    from 'before' the date and ending 'after' the date specified in the pair.
    All those returns series are then aligned to a common index (-before to
    after) and returned as a single DataFrame

    Parameters
    ----------
    factor : pd.DataFrame
        DataFrame with at least date and equity as index, the columns are
        irrelevant
    returns : pd.DataFrame
        A wide form Pandas DataFrame indexed by date with assets in the
        columns. Returns data should span the factor analysis time period
        plus/minus an additional buffer window corresponding to after/before
        period parameters.
    before:
        How many returns to load before factor date
    after:
        How many returns to load after factor date
    cumulative: bool, optional
        Whether or not the given returns are cumulative. If False the given
        returns are assumed to be daily.
    mean_by_date: bool, optional
        If True, compute mean returns for each date and return that
        instead of a return series for each asset
    demean_by: pd.DataFrame, optional
        DataFrame with at least date and equity as index, the columns are
        irrelevant. For each date a list of equities is extracted from
        'demean_by' index and used as universe to compute demeaned mean
        returns (long short portfolio)

    Returns
    -------
    aligned_returns : pd.DataFrame
        Dataframe containing returns series for each factor aligned to the same
        index: -before to after
    """
    if not cumulative:
        returns = returns.apply(cumulative_returns, axis=0)

    all_returns = []

    for timestamp, df in factor.groupby(level='date'):

        equities = df.index.get_level_values('asset')

        try:
            day_zero_index = returns.index.get_loc(timestamp)
        except KeyError:
            continue

        starting_index = max(day_zero_index - before, 0)
        ending_index = min(day_zero_index + after + 1,
                           len(returns.index))

        equities_slice = set(equities)
        if demean_by is not None:
            demean_equities = demean_by.loc[timestamp] \
                .index.get_level_values('asset')
            equities_slice |= set(demean_equities)

        series = returns.loc[returns.index[starting_index:ending_index],
                             equities_slice]
        series.index = range(starting_index - day_zero_index,
                             ending_index - day_zero_index)

        if demean_by is not None:
            mean = series.loc[:, demean_equities].mean(axis=1)
            series = series.loc[:, equities]
            series = series.sub(mean, axis=0)

        if mean_by_date:
            series = series.mean(axis=1)

        all_returns.append(series)

    return pd.concat(all_returns, axis=1)


def average_cumulative_return_by_quantile(factor_data,
                                          returns,
                                          periods_before=10,
                                          periods_after=15,
                                          demeaned=True,
                                          group_adjust=False,
                                          by_group=False):
    """
    Plots average cumulative returns by factor quantiles in the period range
    defined by -periods_before to periods_after

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    returns : pd.DataFrame
        A wide form Pandas DataFrame indexed by date with assets in the
        columns. Returns data should span the factor analysis time period
        plus/minus an additional buffer window corresponding to periods_after/
        periods_before parameters.
    periods_before : int, optional
        How many periods before factor to plot
    periods_after  : int, optional
        How many periods after factor to plot
    demeaned : bool, optional
        Compute demeaned mean returns (long short portfolio)
    group_adjust : bool
        Returns demeaning will occur on the group level (group
        neutral portfolio)
    by_group : bool
        If True, compute cumulative returns separately for each group

    Returns
    -------
    cumulative returns and std deviation : pd.DataFrame
        A MultiIndex DataFrame indexed by quantile (level 0) and mean/std
        (level 1) and the values on the columns in range from
        -periods_before to periods_after
        If by_group=True the index will have an additional 'group' level
        ::
            ---------------------------------------------------
                        |       | -2  | -1  |  0  |  1  | ...
            ---------------------------------------------------
              quantile  |       |     |     |     |     |
            ---------------------------------------------------
                        | mean  |  x  |  x  |  x  |  x  |
                 1      ---------------------------------------
                        | std   |  x  |  x  |  x  |  x  |
            ---------------------------------------------------
                        | mean  |  x  |  x  |  x  |  x  |
                 2      ---------------------------------------
                        | std   |  x  |  x  |  x  |  x  |
            ---------------------------------------------------
                ...     |                 ...
            ---------------------------------------------------
    """

    def cumulative_return_around_event(q_fact, demean_by):
        return common_start_returns(
            q_fact,
            returns,
            periods_before,
            periods_after,
            cumulative=True,
            mean_by_date=True,
            demean_by=demean_by,
        )

    def average_cumulative_return(q_fact, demean_by):
        q_returns = cumulative_return_around_event(q_fact, demean_by)
        q_returns.replace([np.inf, -np.inf], np.nan, inplace=True)

        return pd.DataFrame({'mean': q_returns.mean(skipna=True, axis=1),
                             'std': q_returns.std(skipna=True, axis=1)}).T

    if by_group:
        #
        # Compute quantile cumulative returns separately for each group
        # Deman those returns accordingly to 'group_adjust' and 'demeaned'
        #
        returns_bygroup = []

        for group, g_data in factor_data.groupby('group'):
            g_fq = g_data['factor_quantile']
            if group_adjust:
                demean_by = g_fq  # demeans at group level
            elif demeaned:
                demean_by = factor_data['factor_quantile']  # demean by all
            else:
                demean_by = None
            #
            # Align cumulative return from different dates to the same index
            # then compute mean and std
            #
            avgcumret = g_fq.groupby(g_fq).apply(average_cumulative_return,
                                                 demean_by)
            if len(avgcumret) == 0:
                continue

            avgcumret['group'] = group
            avgcumret.set_index('group', append=True, inplace=True)
            returns_bygroup.append(avgcumret)

        return pd.concat(returns_bygroup, axis=0)

    else:
        #
        # Compute quantile cumulative returns for the full factor_data
        # Align cumulative return from different dates to the same index
        # then compute mean and std
        # Deman those returns accordingly to 'group_adjust' and 'demeaned'
        #
        if group_adjust:
            all_returns = []
            for group, g_data in factor_data.groupby('group'):
                g_fq = g_data['factor_quantile']
                avgcumret = g_fq.groupby(g_fq).apply(
                    cumulative_return_around_event, g_fq
                )
                all_returns.append(avgcumret)
            q_returns = pd.concat(all_returns, axis=1)
            q_returns = pd.DataFrame({'mean': q_returns.mean(axis=1),
                                      'std': q_returns.std(axis=1)})
            return q_returns.unstack(level=1).stack(level=0)
        elif demeaned:
            fq = factor_data['factor_quantile']
            return fq.groupby(fq).apply(average_cumulative_return, fq)
        else:
            fq = factor_data['factor_quantile']
            return fq.groupby(fq).apply(average_cumulative_return, None)


def factor_cumulative_returns(factor_data,
                              period,
                              long_short=True,
                              group_neutral=False,
                              equal_weight=False,
                              quantiles=None,
                              groups=None):
    """
    Simulate a portfolio using the factor in input and returns the cumulative
    returns of the simulated portfolio

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    period : string
        'factor_data' column name corresponding to the 'period' returns to be
        used in the computation of porfolio returns
    long_short : bool, optional
        if True then simulates a dollar neutral long-short portfolio
        - see performance.create_pyfolio_input for more details
    group_neutral : bool, optional
        If True then simulates a group neutral portfolio
        - see performance.create_pyfolio_input for more details
    equal_weight : bool, optional
        Control the assets weights:
        - see performance.create_pyfolio_input for more details
    quantiles: sequence[int], optional
        Use only specific quantiles in the computation. By default all
        quantiles are used
    groups: sequence[string], optional
        Use only specific groups in the computation. By default all groups
        are used

    Returns
    -------
    Cumulative returns series : pd.Series
        Example:
            2015-07-16 09:30:00  -0.012143
            2015-07-16 12:30:00   0.012546
            2015-07-17 09:30:00   0.045350
            2015-07-17 12:30:00   0.065897
            2015-07-20 09:30:00   0.030957
    """
    fwd_ret_cols = utils.get_forward_returns_columns(factor_data.columns)

    if period not in fwd_ret_cols:
        raise ValueError("Period '%s' not found" % period)

    todrop = list(fwd_ret_cols)
    todrop.remove(period)
    portfolio_data = factor_data.drop(todrop, axis=1)

    if quantiles is not None:
        portfolio_data = portfolio_data[portfolio_data['factor_quantile'].isin(
            quantiles)]

    if groups is not None:
        portfolio_data = portfolio_data[portfolio_data['group'].isin(groups)]

    returns = \
        factor_returns(portfolio_data, long_short, group_neutral, equal_weight)

    return cumulative_returns(returns[period], period)


def factor_positions(factor_data,
                     period,
                     long_short=True,
                     group_neutral=False,
                     equal_weight=False,
                     quantiles=None,
                     groups=None):
    """
    Simulate a portfolio using the factor in input and returns the assets
    positions as percentage of the total portfolio.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    period : string
        'factor_data' column name corresponding to the 'period' returns to be
        used in the computation of porfolio returns
    long_short : bool, optional
        if True then simulates a dollar neutral long-short portfolio
        - see performance.create_pyfolio_input for more details
    group_neutral : bool, optional
        If True then simulates a group neutral portfolio
        - see performance.create_pyfolio_input for more details
    equal_weight : bool, optional
        Control the assets weights:
        - see performance.create_pyfolio_input for more details.
    quantiles: sequence[int], optional
        Use only specific quantiles in the computation. By default all
        quantiles are used
    groups: sequence[string], optional
        Use only specific groups in the computation. By default all groups
        are used

    Returns
    -------
    assets positions : pd.DataFrame
        Assets positions series, datetime on index, assets on columns.
        Example:
            index                 'AAPL'         'MSFT'          cash
            2004-01-09 10:30:00   13939.3800     -14012.9930     711.5585
            2004-01-09 15:30:00       0.00       -16012.9930     411.5585
            2004-01-12 10:30:00   14492.6300     -14624.8700       0.0
            2004-01-12 15:30:00   14874.5400     -15841.2500       0.0
            2004-01-13 10:30:00   -13853.2800    13653.6400      -43.6375
    """
    fwd_ret_cols = utils.get_forward_returns_columns(factor_data.columns)

    if period not in fwd_ret_cols:
        raise ValueError("Period '%s' not found" % period)

    todrop = list(fwd_ret_cols)
    todrop.remove(period)
    portfolio_data = factor_data.drop(todrop, axis=1)

    if quantiles is not None:
        portfolio_data = portfolio_data[portfolio_data['factor_quantile'].isin(
            quantiles)]

    if groups is not None:
        portfolio_data = portfolio_data[portfolio_data['group'].isin(groups)]

    weights = \
        factor_weights(portfolio_data, long_short, group_neutral, equal_weight)

    return positions(weights, period)


def create_pyfolio_input(factor_data,
                         period,
                         capital=None,
                         long_short=True,
                         group_neutral=False,
                         equal_weight=False,
                         quantiles=None,
                         groups=None,
                         benchmark_period='1D'):
    """
    Simulate a portfolio using the input factor and returns the portfolio
    performance data properly formatted for Pyfolio analysis.

    For more details on how this portfolio is built see:
    - performance.cumulative_returns (how the portfolio returns are computed)
    - performance.factor_weights (how assets weights are computed)

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    period : string
        'factor_data' column name corresponding to the 'period' returns to be
        used in the computation of porfolio returns
    capital : float, optional
        If set, then compute 'positions' in dollar amount instead of percentage
    long_short : bool, optional
        if True enforce a dollar neutral long-short portfolio: asset weights
        will be computed by demeaning factor values and dividing by the sum of
        their absolute value (achieving gross leverage of 1) which will cause
        the portfolio to hold both long and short positions and the total
        weights of both long and short positions will be equal.
        If False the portfolio weights will be computed dividing the factor
        values and  by the sum of their absolute value (achieving gross
        leverage of 1). Positive factor values will generate long positions and
        negative factor values will produce short positions so that a factor
        with only posive values will result in a long only portfolio.
    group_neutral : bool, optional
        If True simulates a group neutral portfolio: the portfolio weights
        will be computed so that each group will weigh the same.
        if 'long_short' is enabled the factor values demeaning will occur on
        the group level resulting in a dollar neutral, group neutral,
        long-short portfolio.
        If False group information will not be used in weights computation.
    equal_weight : bool, optional
        if True the assets will be equal-weighted. If long_short is True then
        the factor universe will be split in two equal sized groups with the
        top assets in long positions and bottom assets in short positions.
        if False the assets will be factor-weighed, see 'long_short' argument
    quantiles: sequence[int], optional
        Use only specific quantiles in the computation. By default all
        quantiles are used
    groups: sequence[string], optional
        Use only specific groups in the computation. By default all groups
        are used
    benchmark_period : string, optional
        By default benchmark returns are computed as the factor universe mean
        daily returns but 'benchmark_period' allows to choose a 'factor_data'
        column corresponding to the returns to be used in the computation of
        benchmark returns. More generally benchmark returns are computed as the
        factor universe returns traded at 'benchmark_period' frequency, equal
        weighting and long only


    Returns
    -------
     returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - Time series with decimal returns.
         - Example:
            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902

     positions : pd.DataFrame
        Time series of dollar amount (or percentage when 'capital' is not
        provided) invested in each position and cash.
         - Days where stocks are not held can be represented by 0.
         - Non-working capital is labelled 'cash'
         - Example:
            index         'AAPL'         'MSFT'          cash
            2004-01-09    13939.3800     -14012.9930     711.5585
            2004-01-12    14492.6300     -14624.8700     27.1821
            2004-01-13    -13853.2800    13653.6400      -43.6375


     benchmark : pd.Series
        Benchmark returns computed as the factor universe mean daily returns.

    """

    #
    # Build returns:
    # we don't know the frequency at which the factor returns are computed but
    # pyfolio wants daily returns. So we compute the cumulative returns of the
    # factor, then resample it at 1 day frequency and finally compute daily
    # returns
    #
    cumrets = factor_cumulative_returns(factor_data,
                                        period,
                                        long_short,
                                        group_neutral,
                                        equal_weight,
                                        quantiles,
                                        groups)
    cumrets = cumrets.resample('1D').last().fillna(method='ffill')
    returns = cumrets.pct_change().fillna(0)

    #
    # Build positions. As pyfolio asks for daily position we have to resample
    # the positions returned by 'factor_positions' at 1 day frequency and
    # recompute the weights so that the sum of daily weights is 1.0
    #
    positions = factor_positions(factor_data,
                                 period,
                                 long_short,
                                 group_neutral,
                                 equal_weight,
                                 quantiles,
                                 groups)
    positions = positions.resample('1D').sum().fillna(method='ffill')
    positions = positions.div(positions.abs().sum(axis=1), axis=0).fillna(0)
    positions['cash'] = 1. - positions.sum(axis=1)

    # transform percentage positions to dollar positions
    if capital is not None:
        positions = positions.mul(
            cumrets.reindex(positions.index) * capital, axis=0)

    #
    #
    #
    # Build benchmark returns as the factor universe mean returns traded at
    # 'benchmark_period' frequency
    #
    fwd_ret_cols = utils.get_forward_returns_columns(factor_data.columns)
    if benchmark_period in fwd_ret_cols:
        benchmark_data = factor_data.copy()
        # make sure no negative positions
        benchmark_data['factor'] = benchmark_data['factor'].abs()
        benchmark_rets = factor_cumulative_returns(benchmark_data,
                                                   benchmark_period,
                                                   long_short=False,
                                                   group_neutral=False,
                                                   equal_weight=True)
        benchmark_rets = benchmark_rets.resample(
            '1D').last().fillna(method='ffill')
        benchmark_rets = benchmark_rets.pct_change().fillna(0)
        benchmark_rets.name = 'benchmark'
    else:
        benchmark_rets = None

    return returns, positions, benchmark_rets

# -----------------------------------------------------------------------------------------------------------------
#          plotting
# -----------------------------------------------------------------------------------------------------------------

#
# Copyright 2017 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from functools import wraps

from . import utils
from . import performance as perf

DECIMAL_TO_BPS = 10000


def customize(func):
    """
    Decorator to set plotting context and axes style during function call.
    """
    @wraps(func)
    def call_w_context(*args, **kwargs):
        set_context = kwargs.pop('set_context', True)
        if set_context:
            color_palette = sns.color_palette('colorblind')
            with plotting_context(), axes_style(), color_palette:
                sns.despine(left=True)
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return call_w_context


def plotting_context(context='notebook', font_scale=1.5, rc=None):
    """
    Create alphalens default plotting style context.

    Under the hood, calls and returns seaborn.plotting_context() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    context : str, optional
        Name of seaborn context.
    font_scale : float, optional
        Scale font by factor font_scale.
    rc : dict, optional
        Config flags.
        By default, {'lines.linewidth': 1.5}
        is being used and will be added to any
        rc passed in, unless explicitly overriden.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    with alphalens.plotting.plotting_context(font_scale=2):
        alphalens.create_full_tear_sheet(..., set_context=False)

    See also
    --------
    For more information, see seaborn.plotting_context().
    """
    if rc is None:
        rc = {}

    rc_default = {'lines.linewidth': 1.5}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.plotting_context(context=context, font_scale=font_scale, rc=rc)


def axes_style(style='darkgrid', rc=None):
    """Create alphalens default axes style context.

    Under the hood, calls and returns seaborn.axes_style() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    style : str, optional
        Name of seaborn style.
    rc : dict, optional
        Config flags.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    with alphalens.plotting.axes_style(style='whitegrid'):
        alphalens.create_full_tear_sheet(..., set_context=False)

    See also
    --------
    For more information, see seaborn.plotting_context().

    """
    if rc is None:
        rc = {}

    rc_default = {}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.axes_style(style=style, rc=rc)


def plot_returns_table(alpha_beta,
                       mean_ret_quantile,
                       mean_ret_spread_quantile):
    returns_table = pd.DataFrame()
    returns_table = returns_table.append(alpha_beta)
    returns_table.loc["Mean Period Wise Return Top Quantile (bps)"] = \
        mean_ret_quantile.iloc[-1] * DECIMAL_TO_BPS
    returns_table.loc["Mean Period Wise Return Bottom Quantile (bps)"] = \
        mean_ret_quantile.iloc[0] * DECIMAL_TO_BPS
    returns_table.loc["Mean Period Wise Spread (bps)"] = \
        mean_ret_spread_quantile.mean() * DECIMAL_TO_BPS

    print("Returns Analysis")
    utils.print_table(returns_table.apply(lambda x: x.round(3)))


def plot_turnover_table(autocorrelation_data, quantile_turnover):
    turnover_table = pd.DataFrame()
    for period in sorted(quantile_turnover.keys()):
        for quantile, p_data in quantile_turnover[period].iteritems():
            turnover_table.loc["Quantile {} Mean Turnover ".format(quantile),
                               "{}D".format(period)] = p_data.mean()
    auto_corr = pd.DataFrame()
    for period, p_data in autocorrelation_data.iteritems():
        auto_corr.loc["Mean Factor Rank Autocorrelation",
                      "{}D".format(period)] = p_data.mean()

    print("Turnover Analysis")
    utils.print_table(turnover_table.apply(lambda x: x.round(3)))
    utils.print_table(auto_corr.apply(lambda x: x.round(3)))


def plot_information_table(ic_data):
    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["Risk-Adjusted IC"] = \
        ic_data.mean() / ic_data.std()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)

    print("Information Analysis")
    utils.print_table(ic_summary_table.apply(lambda x: x.round(3)).T)


def plot_quantile_statistics_table(factor_data):
    quantile_stats = factor_data.groupby('factor_quantile') \
        .agg(['min', 'max', 'mean', 'std', 'count'])['factor']
    quantile_stats['count %'] = quantile_stats['count'] \
        / quantile_stats['count'].sum() * 100.

    print("Quantiles Statistics")
    utils.print_table(quantile_stats)


def plot_ic_ts(ic, ax=None):
    """
    Plots Spearman Rank Information Coefficient and IC moving
    average for a given factor.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    ic = ic.copy()

    num_plots = len(ic.columns)
    if ax is None:
        f, ax = plt.subplots(num_plots, 1, figsize=(18, num_plots * 7))
        ax = np.asarray([ax]).flatten()

    ymin, ymax = (None, None)
    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        ic.plot(alpha=0.7, ax=a, lw=0.7, color='steelblue')
        ic.rolling(window=22).mean().plot(
            ax=a,
            color='forestgreen',
            lw=2,
            alpha=0.8
        )

        a.set(ylabel='IC', xlabel="")
        a.set_title(
            "{} Period Forward Return Information Coefficient (IC)"
            .format(period_num))
        a.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)
        a.legend(['IC', '1 month moving avg'], loc='upper right')
        a.text(.05, .95, "Mean %.3f \n Std. %.3f" % (ic.mean(), ic.std()),
               fontsize=16,
               bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
               transform=a.transAxes,
               verticalalignment='top')

        curr_ymin, curr_ymax = a.get_ylim()
        ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
        ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)

    for a in ax:
        a.set_ylim([ymin, ymax])

    return ax


def plot_ic_hist(ic, ax=None):
    """
    Plots Spearman Rank Information Coefficient histogram for a given factor.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    ic = ic.copy()

    num_plots = len(ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        sns.distplot(ic.replace(np.nan, 0.), norm_hist=True, ax=a)
        a.set(title="%s Period IC" % period_num, xlabel='IC')
        a.set_xlim([-1, 1])
        a.text(.05, .95, "Mean %.3f \n Std. %.3f" % (ic.mean(), ic.std()),
               fontsize=16,
               bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
               transform=a.transAxes,
               verticalalignment='top')
        a.axvline(ic.mean(), color='w', linestyle='dashed', linewidth=2)

    if num_plots < len(ax):
        ax[-1].set_visible(False)

    return ax


def plot_ic_qq(ic, theoretical_dist=stats.norm, ax=None):
    """
    Plots Spearman Rank Information Coefficient "Q-Q" plot relative to
    a theoretical distribution.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    theoretical_dist : scipy.stats._continuous_distns
        Continuous distribution generator. scipy.stats.norm and
        scipy.stats.t are popular options.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    ic = ic.copy()

    num_plots = len(ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    if isinstance(theoretical_dist, stats.norm.__class__):
        dist_name = 'Normal'
    elif isinstance(theoretical_dist, stats.t.__class__):
        dist_name = 'T'
    else:
        dist_name = 'Theoretical'

    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        sm.qqplot(ic.replace(np.nan, 0.).values, theoretical_dist, fit=True,
                  line='45', ax=a)
        a.set(title="{} Period IC {} Dist. Q-Q".format(
              period_num, dist_name),
              ylabel='Observed Quantile',
              xlabel='{} Distribution Quantile'.format(dist_name))

    return ax


def plot_quantile_returns_bar(mean_ret_by_q,
                              by_group=False,
                              ylim_percentiles=None,
                              ax=None):
    """
    Plots mean period wise returns for factor quantiles.

    Parameters
    ----------
    mean_ret_by_q : pd.DataFrame
        DataFrame with quantile, (group) and mean period wise return values.
    by_group : bool
        Disaggregated figures by group.
    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    mean_ret_by_q = mean_ret_by_q.copy()

    if ylim_percentiles is not None:
        ymin = (np.nanpercentile(mean_ret_by_q.values,
                                 ylim_percentiles[0]) * DECIMAL_TO_BPS)
        ymax = (np.nanpercentile(mean_ret_by_q.values,
                                 ylim_percentiles[1]) * DECIMAL_TO_BPS)
    else:
        ymin = None
        ymax = None

    if by_group:
        num_group = len(
            mean_ret_by_q.index.get_level_values('group').unique())

        if ax is None:
            v_spaces = ((num_group - 1) // 2) + 1
            f, ax = plt.subplots(v_spaces, 2, sharex=False,
                                 sharey=True, figsize=(18, 6 * v_spaces))
            ax = ax.flatten()

        for a, (sc, cor) in zip(ax, mean_ret_by_q.groupby(level='group')):
            (cor.xs(sc, level='group')
                .multiply(DECIMAL_TO_BPS)
                .plot(kind='bar', title=sc, ax=a))

            a.set(xlabel='', ylabel='Mean Return (bps)',
                  ylim=(ymin, ymax))

        if num_group < len(ax):
            ax[-1].set_visible(False)

        return ax

    else:
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(18, 6))

        (mean_ret_by_q.multiply(DECIMAL_TO_BPS)
            .plot(kind='bar',
                  title="Mean Period Wise Return By Factor Quantile", ax=ax))
        ax.set(xlabel='', ylabel='Mean Return (bps)',
               ylim=(ymin, ymax))

        return ax


def plot_quantile_returns_violin(return_by_q,
                                 ylim_percentiles=None,
                                 ax=None):
    """
    Plots a violin box plot of period wise returns for factor quantiles.

    Parameters
    ----------
    return_by_q : pd.DataFrame - MultiIndex
        DataFrame with date and quantile as rows MultiIndex,
        forward return windows as columns, returns as values.
    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    return_by_q = return_by_q.copy()

    if ylim_percentiles is not None:
        ymin = (np.nanpercentile(return_by_q.values,
                                 ylim_percentiles[0]) * DECIMAL_TO_BPS)
        ymax = (np.nanpercentile(return_by_q.values,
                                 ylim_percentiles[1]) * DECIMAL_TO_BPS)
    else:
        ymin = None
        ymax = None

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    unstacked_dr = (return_by_q
                    .multiply(DECIMAL_TO_BPS))
    unstacked_dr.columns = unstacked_dr.columns.set_names('forward_periods')
    unstacked_dr = unstacked_dr.stack()
    unstacked_dr.name = 'return'
    unstacked_dr = unstacked_dr.reset_index()

    sns.violinplot(data=unstacked_dr,
                   x='factor_quantile',
                   hue='forward_periods',
                   y='return',
                   orient='v',
                   cut=0,
                   inner='quartile',
                   ax=ax)
    ax.set(xlabel='', ylabel='Return (bps)',
           title="Period Wise Return By Factor Quantile",
           ylim=(ymin, ymax))

    ax.axhline(0.0, linestyle='-', color='black', lw=0.7, alpha=0.6)

    return ax


def plot_mean_quantile_returns_spread_time_series(mean_returns_spread,
                                                  std_err=None,
                                                  bandwidth=1,
                                                  ax=None):
    """
    Plots mean period wise returns for factor quantiles.

    Parameters
    ----------
    mean_returns_spread : pd.Series
        Series with difference between quantile mean returns by period.
    std_err : pd.Series
        Series with standard error of difference between quantile
        mean returns each period.
    bandwidth : float
        Width of displayed error bands in standard deviations.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if isinstance(mean_returns_spread, pd.DataFrame):
        if ax is None:
            ax = [None for a in mean_returns_spread.columns]

        ymin, ymax = (None, None)
        for (i, a), (name, fr_column) in zip(enumerate(ax),
                                             mean_returns_spread.iteritems()):
            stdn = None if std_err is None else std_err[name]
            a = plot_mean_quantile_returns_spread_time_series(fr_column,
                                                              std_err=stdn,
                                                              ax=a)
            ax[i] = a
            curr_ymin, curr_ymax = a.get_ylim()
            ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
            ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)

        for a in ax:
            a.set_ylim([ymin, ymax])

        return ax

    if mean_returns_spread.isnull().all():
        return ax

    periods = mean_returns_spread.name
    title = ('Top Minus Bottom Quantile Mean Return ({} Period Forward Return)'
             .format(periods if periods is not None else ""))

    if ax is None:
        f, ax = plt.subplots(figsize=(18, 6))

    mean_returns_spread_bps = mean_returns_spread * DECIMAL_TO_BPS

    mean_returns_spread_bps.plot(alpha=0.4, ax=ax, lw=0.7, color='forestgreen')
    mean_returns_spread_bps.rolling(window=22).mean().plot(
        color='orangered',
        alpha=0.7,
        ax=ax
    )
    ax.legend(['mean returns spread', '1 month moving avg'], loc='upper right')

    if std_err is not None:
        std_err_bps = std_err * DECIMAL_TO_BPS
        upper = mean_returns_spread_bps.values + (std_err_bps * bandwidth)
        lower = mean_returns_spread_bps.values - (std_err_bps * bandwidth)
        ax.fill_between(mean_returns_spread.index,
                        lower,
                        upper,
                        alpha=0.3,
                        color='steelblue')

    ylim = np.nanpercentile(abs(mean_returns_spread_bps.values), 95)
    ax.set(ylabel='Difference In Quantile Mean Return (bps)',
           xlabel='',
           title=title,
           ylim=(-ylim, ylim))
    ax.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)

    return ax


def plot_ic_by_group(ic_group, ax=None):
    """
    Plots Spearman Rank Information Coefficient for a given factor over
    provided forward returns. Separates by group.

    Parameters
    ----------
    ic_group : pd.DataFrame
        group-wise mean period wise returns.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))
    ic_group.plot(kind='bar', ax=ax)

    ax.set(title="Information Coefficient By Group", xlabel="")
    ax.set_xticklabels(ic_group.index, rotation=45)

    return ax


def plot_factor_rank_auto_correlation(factor_autocorrelation,
                                      period=1,
                                      ax=None):
    """
    Plots factor rank autocorrelation over time.
    See factor_rank_autocorrelation for more details.

    Parameters
    ----------
    factor_autocorrelation : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation
        of factor values.
    period: int, optional
        Period over which the autocorrelation is calculated
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    factor_autocorrelation.plot(title='{}D Period Factor Rank Autocorrelation'
                                .format(period), ax=ax)
    ax.set(ylabel='Autocorrelation Coefficient', xlabel='')
    ax.axhline(0.0, linestyle='-', color='black', lw=1)
    ax.text(.05, .95, "Mean %.3f" % factor_autocorrelation.mean(),
            fontsize=16,
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
            transform=ax.transAxes,
            verticalalignment='top')

    return ax


def plot_top_bottom_quantile_turnover(quantile_turnover, period=1, ax=None):
    """
    Plots period wise top and bottom quantile factor turnover.

    Parameters
    ----------
    quantile_turnover: pd.Dataframe
        Quantile turnover (each DataFrame column a quantile).
    period: int, optional
        Period over which to calculate the turnover.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    max_quantile = quantile_turnover.columns.max()
    min_quantile = quantile_turnover.columns.min()
    turnover = pd.DataFrame()
    turnover['top quantile turnover'] = quantile_turnover[max_quantile]
    turnover['bottom quantile turnover'] = quantile_turnover[min_quantile]
    turnover.plot(title='{}D Period Top and Bottom Quantile Turnover'
                  .format(period), ax=ax, alpha=0.6, lw=0.8)
    ax.set(ylabel='Proportion Of Names New To Quantile', xlabel="")

    return ax


def plot_monthly_ic_heatmap(mean_monthly_ic, ax=None):
    """
    Plots a heatmap of the information coefficient or returns by month.

    Parameters
    ----------
    mean_monthly_ic : pd.DataFrame
        The mean monthly IC for N periods forward.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    mean_monthly_ic = mean_monthly_ic.copy()

    num_plots = len(mean_monthly_ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    new_index_year = []
    new_index_month = []
    for date in mean_monthly_ic.index:
        new_index_year.append(date.year)
        new_index_month.append(date.month)

    mean_monthly_ic.index = pd.MultiIndex.from_arrays(
        [new_index_year, new_index_month],
        names=["year", "month"])

    for a, (periods_num, ic) in zip(ax, mean_monthly_ic.iteritems()):

        sns.heatmap(
            ic.unstack(),
            annot=True,
            alpha=1.0,
            center=0.0,
            annot_kws={"size": 7},
            linewidths=0.01,
            linecolor='white',
            cmap=cm.coolwarm_r,
            cbar=False,
            ax=a)
        a.set(ylabel='', xlabel='')

        a.set_title("Monthly Mean {} Period IC".format(periods_num))

    if num_plots < len(ax):
        ax[-1].set_visible(False)

    return ax


def plot_cumulative_returns(factor_returns,
                            period,
                            freq=None,
                            title=None,
                            ax=None):
    """
    Plots the cumulative returns of the returns series passed in.

    Parameters
    ----------
    factor_returns : pd.Series
        Period wise returns of dollar neutral portfolio weighted by factor
        value.
    period : pandas.Timedelta or string
        Length of period for which the returns are computed (e.g. 1 day)
        if 'period' is a string it must follow pandas.Timedelta constructor
        format (e.g. '1 days', '1D', '30m', '3h', '1D1h', etc)
    freq : pandas DateOffset
        Used to specify a particular trading calendar e.g. BusinessDay or Day
        Usually this is inferred from utils.infer_trading_calendar, which is
        called by either get_clean_factor_and_forward_returns or
        compute_forward_returns
    title: string, optional
        Custom title
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    factor_returns = perf.cumulative_returns(factor_returns)

    factor_returns.plot(ax=ax, lw=3, color='forestgreen', alpha=0.6)
    ax.set(ylabel='Cumulative Returns',
           title=("Portfolio Cumulative Return ({} Fwd Period)".format(period)
                  if title is None else title),
           xlabel='')
    ax.axhline(1.0, linestyle='-', color='black', lw=1)

    return ax


def plot_cumulative_returns_by_quantile(quantile_returns,
                                        period,
                                        freq=None,
                                        ax=None):
    """
    Plots the cumulative returns of various factor quantiles.

    Parameters
    ----------
    quantile_returns : pd.DataFrame
        Returns by factor quantile
    period : pandas.Timedelta or string
        Length of period for which the returns are computed (e.g. 1 day)
        if 'period' is a string it must follow pandas.Timedelta constructor
        format (e.g. '1 days', '1D', '30m', '3h', '1D1h', etc)
    freq : pandas DateOffset
        Used to specify a particular trading calendar e.g. BusinessDay or Day
        Usually this is inferred from utils.infer_trading_calendar, which is
        called by either get_clean_factor_and_forward_returns or
        compute_forward_returns
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
    """

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    ret_wide = quantile_returns.unstack('factor_quantile')

    cum_ret = ret_wide.apply(perf.cumulative_returns)

    cum_ret = cum_ret.loc[:, ::-1]  # we want negative quantiles as 'red'

    cum_ret.plot(lw=2, ax=ax, cmap=cm.coolwarm)
    ax.legend()
    ymin, ymax = cum_ret.min().min(), cum_ret.max().max()
    ax.set(ylabel='Log Cumulative Returns',
           title='''Cumulative Return by Quantile
                    ({} Period Forward Return)'''.format(period),
           xlabel='',
           yscale='symlog',
           yticks=np.linspace(ymin, ymax, 5),
           ylim=(ymin, ymax))

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.axhline(1.0, linestyle='-', color='black', lw=1)

    return ax


def plot_quantile_average_cumulative_return(avg_cumulative_returns,
                                            by_quantile=False,
                                            std_bar=False,
                                            title=None,
                                            ax=None):
    """
    Plots sector-wise mean daily returns for factor quantiles
    across provided forward price movement columns.

    Parameters
    ----------
    avg_cumulative_returns: pd.Dataframe
        The format is the one returned by
        performance.average_cumulative_return_by_quantile
    by_quantile : boolean, optional
        Disaggregated figures by quantile (useful to clearly see std dev bars)
    std_bar : boolean, optional
        Plot standard deviation plot
    title: string, optional
        Custom title
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
    """

    avg_cumulative_returns = avg_cumulative_returns.multiply(DECIMAL_TO_BPS)
    quantiles = len(avg_cumulative_returns.index.levels[0].unique())
    palette = [cm.coolwarm(i) for i in np.linspace(0, 1, quantiles)]
    palette = palette[::-1]  # we want negative quantiles as 'red'

    if by_quantile:

        if ax is None:
            v_spaces = ((quantiles - 1) // 2) + 1
            f, ax = plt.subplots(v_spaces, 2, sharex=False,
                                 sharey=False, figsize=(18, 6 * v_spaces))
            ax = ax.flatten()

        for i, (quantile, q_ret) in enumerate(avg_cumulative_returns
                                              .groupby(level='factor_quantile')
                                              ):

            mean = q_ret.loc[(quantile, 'mean')]
            mean.name = 'Quantile ' + str(quantile)
            mean.plot(ax=ax[i], color=palette[i])
            ax[i].set_ylabel('Mean Return (bps)')

            if std_bar:
                std = q_ret.loc[(quantile, 'std')]
                ax[i].errorbar(std.index, mean, yerr=std,
                               fmt='none', ecolor=palette[i], label='none')

            ax[i].axvline(x=0, color='k', linestyle='--')
            ax[i].legend()
            i += 1

    else:

        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(18, 6))

        for i, (quantile, q_ret) in enumerate(avg_cumulative_returns
                                              .groupby(level='factor_quantile')
                                              ):

            mean = q_ret.loc[(quantile, 'mean')]
            mean.name = 'Quantile ' + str(quantile)
            mean.plot(ax=ax, color=palette[i])

            if std_bar:
                std = q_ret.loc[(quantile, 'std')]
                ax.errorbar(std.index, mean, yerr=std,
                            fmt='none', ecolor=palette[i], label='none')
            i += 1

        ax.axvline(x=0, color='k', linestyle='--')
        ax.legend()
        ax.set(ylabel='Mean Return (bps)',
               title=("Average Cumulative Returns by Quantile"
                      if title is None else title),
               xlabel='Periods')

    return ax


def plot_events_distribution(events, num_bars=50, ax=None):
    """
    Plots the distribution of events in time.

    Parameters
    ----------
    events : pd.Series
        A pd.Series whose index contains at least 'date' level.
    num_bars : integer, optional
        Number of bars to plot
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
    """

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    start = events.index.get_level_values('date').min()
    end = events.index.get_level_values('date').max()
    group_interval = (end - start) / num_bars
    grouper = pd.Grouper(level='date', freq=group_interval)
    events.groupby(grouper).count().plot(kind="bar", grid=False, ax=ax)
    ax.set(ylabel='Number of events',
           title='Distribution of events in time',
           xlabel='Date')

    return ax


# -----------------------------------------------------------------------------------------------------------------
#          tears
# -----------------------------------------------------------------------------------------------------------------

#
# Copyright 2017 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import warnings

from . import plotting
from . import performance as perf
from . import utils


class GridFigure(object):
    """
    It makes life easier with grid plots
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure(figsize=(14, rows * 7))
        self.gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.3)
        self.curr_row = 0
        self.curr_col = 0

    def next_row(self):
        if self.curr_col != 0:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, :])
        self.curr_row += 1
        return subplt

    def next_cell(self):
        if self.curr_col >= self.cols:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, self.curr_col])
        self.curr_col += 1
        return subplt

    def close(self):
        plt.close(self.fig)
        self.fig = None
        self.gs = None


@plotting.customize
def create_summary_tear_sheet(
    factor_data, long_short=True, group_neutral=False
):
    """
    Creates a small summary tear sheet with returns, information, and turnover
    analysis.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio? if so, then
        mean quantile returns will be demeaned across the factor universe.
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
    """

    # Returns Analysis
    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
    )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )

    compstd_quant_daily = std_quant_daily.apply(
        utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    )

    alpha_beta = perf.factor_alpha_beta(
        factor_data, demeaned=long_short, group_adjust=group_neutral
    )

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        factor_data["factor_quantile"].max(),
        factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )

    periods = utils.get_forward_returns_columns(factor_data.columns)
    periods = list(map(lambda p: pd.Timedelta(p).days, periods))

    fr_cols = len(periods)
    vertical_sections = 2 + fr_cols * 3
    gf = GridFigure(rows=vertical_sections, cols=1)

    plotting.plot_quantile_statistics_table(factor_data)

    plotting.plot_returns_table(
        alpha_beta, mean_quant_rateret, mean_ret_spread_quant
    )

    plotting.plot_quantile_returns_bar(
        mean_quant_rateret,
        by_group=False,
        ylim_percentiles=None,
        ax=gf.next_row(),
    )

    # Information Analysis
    ic = perf.factor_information_coefficient(factor_data)
    plotting.plot_information_table(ic)

    # Turnover Analysis
    quantile_factor = factor_data["factor_quantile"]

    quantile_turnover = {
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in range(1, int(quantile_factor.max()) + 1)
            ],
            axis=1,
        )
        for p in periods
    }

    autocorrelation = pd.concat(
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in periods
        ],
        axis=1,
    )

    plotting.plot_turnover_table(autocorrelation, quantile_turnover)

    plt.show()
    gf.close()


@plotting.customize
def create_returns_tear_sheet(
    factor_data, long_short=True, group_neutral=False, by_group=False
):
    """
    Creates a tear sheet for returns analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio? if so, then
        mean quantile returns will be demeaned across the factor universe.
        Additionally factor values will be demeaned across the factor universe
        when factor weighting the portfolio for cumulative returns plots
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
        Additionally each group will weight the same in cumulative returns
        plots
    by_group : bool
        If True, display graphs separately for each group.
    """

    factor_returns = perf.factor_returns(
        factor_data, long_short, group_neutral
    )

    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
    )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )

    compstd_quant_daily = std_quant_daily.apply(
        utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    )

    alpha_beta = perf.factor_alpha_beta(
        factor_data, factor_returns, long_short, group_neutral
    )

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        factor_data["factor_quantile"].max(),
        factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )

    fr_cols = len(factor_returns.columns)
    vertical_sections = 2 + fr_cols * 3
    gf = GridFigure(rows=vertical_sections, cols=1)

    plotting.plot_returns_table(
        alpha_beta, mean_quant_rateret, mean_ret_spread_quant
    )

    plotting.plot_quantile_returns_bar(
        mean_quant_rateret,
        by_group=False,
        ylim_percentiles=None,
        ax=gf.next_row(),
    )

    plotting.plot_quantile_returns_violin(
        mean_quant_rateret_bydate, ylim_percentiles=(1, 99), ax=gf.next_row()
    )

    trading_calendar = factor_data.index.levels[0].freq
    if trading_calendar is None:
        trading_calendar = pd.tseries.offsets.BDay()
        warnings.warn(
            "'freq' not set in factor_data index: assuming business day",
            UserWarning,
        )

    # Compute cumulative returns from daily simple returns, if '1D'
    # returns are provided.
    if "1D" in factor_returns:
        title = (
            "Factor Weighted "
            + ("Group Neutral " if group_neutral else "")
            + ("Long/Short " if long_short else "")
            + "Portfolio Cumulative Return (1D Period)"
        )

        plotting.plot_cumulative_returns(
            factor_returns["1D"], period="1D", title=title, ax=gf.next_row()
        )

        plotting.plot_cumulative_returns_by_quantile(
            mean_quant_ret_bydate["1D"], period="1D", ax=gf.next_row()
        )

    ax_mean_quantile_returns_spread_ts = [
        gf.next_row() for x in range(fr_cols)
    ]
    plotting.plot_mean_quantile_returns_spread_time_series(
        mean_ret_spread_quant,
        std_err=std_spread_quant,
        bandwidth=0.5,
        ax=ax_mean_quantile_returns_spread_ts,
    )

    plt.show()
    gf.close()

    if by_group:
        (
            mean_return_quantile_group,
            mean_return_quantile_group_std_err,
        ) = perf.mean_return_by_quantile(
            factor_data,
            by_date=False,
            by_group=True,
            demeaned=long_short,
            group_adjust=group_neutral,
        )

        mean_quant_rateret_group = mean_return_quantile_group.apply(
            utils.rate_of_return,
            axis=0,
            base_period=mean_return_quantile_group.columns[0],
        )

        num_groups = len(
            mean_quant_rateret_group.index.get_level_values("group").unique()
        )

        vertical_sections = 1 + (((num_groups - 1) // 2) + 1)
        gf = GridFigure(rows=vertical_sections, cols=2)

        ax_quantile_returns_bar_by_group = [
            gf.next_cell() for _ in range(num_groups)
        ]
        plotting.plot_quantile_returns_bar(
            mean_quant_rateret_group,
            by_group=True,
            ylim_percentiles=(5, 95),
            ax=ax_quantile_returns_bar_by_group,
        )
        plt.show()
        gf.close()


@plotting.customize
def create_information_tear_sheet(
    factor_data, group_neutral=False, by_group=False
):
    """
    Creates a tear sheet for information analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    group_neutral : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, display graphs separately for each group.
    """

    ic = perf.factor_information_coefficient(factor_data, group_neutral)

    plotting.plot_information_table(ic)

    columns_wide = 2
    fr_cols = len(ic.columns)
    rows_when_wide = ((fr_cols - 1) // columns_wide) + 1
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    ax_ic_ts = [gf.next_row() for _ in range(fr_cols)]
    plotting.plot_ic_ts(ic, ax=ax_ic_ts)

    ax_ic_hqq = [gf.next_cell() for _ in range(fr_cols * 2)]
    plotting.plot_ic_hist(ic, ax=ax_ic_hqq[::2])
    plotting.plot_ic_qq(ic, ax=ax_ic_hqq[1::2])

    if not by_group:

        mean_monthly_ic = perf.mean_information_coefficient(
            factor_data,
            group_adjust=group_neutral,
            by_group=False,
            by_time="M",
        )
        ax_monthly_ic_heatmap = [gf.next_cell() for x in range(fr_cols)]
        plotting.plot_monthly_ic_heatmap(
            mean_monthly_ic, ax=ax_monthly_ic_heatmap
        )

    if by_group:
        mean_group_ic = perf.mean_information_coefficient(
            factor_data, group_adjust=group_neutral, by_group=True
        )

        plotting.plot_ic_by_group(mean_group_ic, ax=gf.next_row())

    plt.show()
    gf.close()


@plotting.customize
def create_turnover_tear_sheet(factor_data, turnover_periods=None):
    """
    Creates a tear sheet for analyzing the turnover properties of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    turnover_periods : sequence[string], optional
        Periods to compute turnover analysis on. By default periods in
        'factor_data' are used but custom periods can provided instead. This
        can be useful when periods in 'factor_data' are not multiples of the
        frequency at which factor values are computed i.e. the periods
        are 2h and 4h and the factor is computed daily and so values like
        ['1D', '2D'] could be used instead
    """

    if turnover_periods is None:
        input_periods = utils.get_forward_returns_columns(
            factor_data.columns, require_exact_day_multiple=True,
        ).get_values()
        turnover_periods = utils.timedelta_strings_to_integers(input_periods)
    else:
        turnover_periods = utils.timedelta_strings_to_integers(
            turnover_periods,
        )

    quantile_factor = factor_data["factor_quantile"]

    quantile_turnover = {
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in quantile_factor.sort_values().unique().tolist()
            ],
            axis=1,
        )
        for p in turnover_periods
    }

    autocorrelation = pd.concat(
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in turnover_periods
        ],
        axis=1,
    )

    plotting.plot_turnover_table(autocorrelation, quantile_turnover)

    fr_cols = len(turnover_periods)
    columns_wide = 1
    rows_when_wide = ((fr_cols - 1) // 1) + 1
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    for period in turnover_periods:
        if quantile_turnover[period].isnull().all().all():
            continue
        plotting.plot_top_bottom_quantile_turnover(
            quantile_turnover[period], period=period, ax=gf.next_row()
        )

    for period in autocorrelation:
        if autocorrelation[period].isnull().all():
            continue
        plotting.plot_factor_rank_auto_correlation(
            autocorrelation[period], period=period, ax=gf.next_row()
        )

    plt.show()
    gf.close()


@plotting.customize
def create_full_tear_sheet(factor_data,
                           long_short=True,
                           group_neutral=False,
                           by_group=False):
    """
    Creates a full tear sheet for analysis and evaluating single
    return predicting (alpha) factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio?
        - See tears.create_returns_tear_sheet for details on how this flag
        affects returns analysis
    group_neutral : bool
        Should this computation happen on a group neutral portfolio?
        - See tears.create_returns_tear_sheet for details on how this flag
        affects returns analysis
        - See tears.create_information_tear_sheet for details on how this
        flag affects information analysis
    by_group : bool
        If True, display graphs separately for each group.
    """

    plotting.plot_quantile_statistics_table(factor_data)
    create_returns_tear_sheet(
        factor_data, long_short, group_neutral, by_group, set_context=False
    )
    create_information_tear_sheet(
        factor_data, group_neutral, by_group, set_context=False
    )
    create_turnover_tear_sheet(factor_data, set_context=False)


@plotting.customize
def create_event_returns_tear_sheet(factor_data,
                                    returns,
                                    avgretplot=(5, 15),
                                    long_short=True,
                                    group_neutral=False,
                                    std_bar=True,
                                    by_group=False):
    """
    Creates a tear sheet to view the average cumulative returns for a
    factor within a window (pre and post event).

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, the factor
        quantile/bin that factor value belongs to and (optionally) the group
        the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    returns : pd.DataFrame
        A DataFrame indexed by date with assets in the columns containing daily
        returns.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    avgretplot: tuple (int, int) - (before, after)
        If not None, plot quantile average cumulative returns
    long_short : bool
        Should this computation happen on a long short portfolio? if so then
        factor returns will be demeaned across the factor universe
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
    std_bar : boolean, optional
        Show plots with standard deviation bars, one for each quantile
    by_group : bool
        If True, display graphs separately for each group.
    """

    before, after = avgretplot

    avg_cumulative_returns = perf.average_cumulative_return_by_quantile(
        factor_data,
        returns,
        periods_before=before,
        periods_after=after,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    num_quantiles = int(factor_data["factor_quantile"].max())

    vertical_sections = 1
    if std_bar:
        vertical_sections += ((num_quantiles - 1) // 2) + 1
    cols = 2 if num_quantiles != 1 else 1
    gf = GridFigure(rows=vertical_sections, cols=cols)
    plotting.plot_quantile_average_cumulative_return(
        avg_cumulative_returns,
        by_quantile=False,
        std_bar=False,
        ax=gf.next_row(),
    )
    if std_bar:
        ax_avg_cumulative_returns_by_q = [
            gf.next_cell() for _ in range(num_quantiles)
        ]
        plotting.plot_quantile_average_cumulative_return(
            avg_cumulative_returns,
            by_quantile=True,
            std_bar=True,
            ax=ax_avg_cumulative_returns_by_q,
        )

    plt.show()
    gf.close()

    if by_group:
        groups = factor_data["group"].unique()
        num_groups = len(groups)
        vertical_sections = ((num_groups - 1) // 2) + 1
        gf = GridFigure(rows=vertical_sections, cols=2)

        avg_cumret_by_group = perf.average_cumulative_return_by_quantile(
            factor_data,
            returns,
            periods_before=before,
            periods_after=after,
            demeaned=long_short,
            group_adjust=group_neutral,
            by_group=True,
        )

        for group, avg_cumret in avg_cumret_by_group.groupby(level="group"):
            avg_cumret.index = avg_cumret.index.droplevel("group")
            plotting.plot_quantile_average_cumulative_return(
                avg_cumret,
                by_quantile=False,
                std_bar=False,
                title=group,
                ax=gf.next_cell(),
            )

        plt.show()
        gf.close()


@plotting.customize
def create_event_study_tear_sheet(factor_data,
                                  returns,
                                  avgretplot=(5, 15),
                                  rate_of_ret=True,
                                  n_bars=50):
    """
    Creates an event study tear sheet for analysis of a specific event.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single event, forward returns for each
        period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
    returns : pd.DataFrame, required only if 'avgretplot' is provided
        A DataFrame indexed by date with assets in the columns containing daily
        returns.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    avgretplot: tuple (int, int) - (before, after), optional
        If not None, plot event style average cumulative returns within a
        window (pre and post event).
    rate_of_ret : bool, optional
        Display rate of return instead of simple return in 'Mean Period Wise
        Return By Factor Quantile' and 'Period Wise Return By Factor Quantile'
        plots
    n_bars : int, optional
        Number of bars in event distribution plot
    """

    long_short = False

    plotting.plot_quantile_statistics_table(factor_data)

    gf = GridFigure(rows=1, cols=1)
    plotting.plot_events_distribution(
        events=factor_data["factor"], num_bars=n_bars, ax=gf.next_row()
    )
    plt.show()
    gf.close()

    if returns is not None and avgretplot is not None:

        create_event_returns_tear_sheet(
            factor_data=factor_data,
            returns=returns,
            avgretplot=avgretplot,
            long_short=long_short,
            group_neutral=False,
            std_bar=True,
            by_group=False,
        )

    factor_returns = perf.factor_returns(
        factor_data, demeaned=False, equal_weight=True
    )

    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data, by_group=False, demeaned=long_short
    )
    if rate_of_ret:
        mean_quant_ret = mean_quant_ret.apply(
            utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
        )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data, by_date=True, by_group=False, demeaned=long_short
    )
    if rate_of_ret:
        mean_quant_ret_bydate = mean_quant_ret_bydate.apply(
            utils.rate_of_return,
            axis=0,
            base_period=mean_quant_ret_bydate.columns[0],
        )

    fr_cols = len(factor_returns.columns)
    vertical_sections = 2 + fr_cols * 1
    gf = GridFigure(rows=vertical_sections + 1, cols=1)

    plotting.plot_quantile_returns_bar(
        mean_quant_ret, by_group=False, ylim_percentiles=None, ax=gf.next_row()
    )

    plotting.plot_quantile_returns_violin(
        mean_quant_ret_bydate, ylim_percentiles=(1, 99), ax=gf.next_row()
    )

    trading_calendar = factor_data.index.levels[0].freq
    if trading_calendar is None:
        trading_calendar = pd.tseries.offsets.BDay()
        warnings.warn(
            "'freq' not set in factor_data index: assuming business day",
            UserWarning,
        )

    plt.show()
    gf.close()
