# -*- coding: utf-8 -*-
"""
@Time    : 2022/6/8
@Author  : congcong009
@Email   : congcong009@foxmail.com
"""

import datetime as dt
import numpy as np
import pandas as pd
import empyrical as ep
import pickle
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from jqdata import *
from jqfactor import calc_factors, Factor, analyze_factor
from typing import (List, Tuple, Dict, Callable, Union)
# from kuanke.user_space_api import *

"""
============================================================================================================================================================
                                                                封装的 jqdata 数据处理
============================================================================================================================================================
"""


# 获取行业
def get_stock_ind(securities,
                  watch_date,
                  level='sw_l1',
                  method='industry_code'
                  ):
    """
    获取行业

    Parameters
    -----------
    securities: list
        股票列表
    watch_date: str
        查询日期
    level: str
        查询股票所属行业级别
    method: str
        返回行业名称or代码

    Returns
    -----------
    indusrty_ser: pd.Series
        行业代码序列
    """
    # 查询股票所属行业，返回一个dict， key是标的代码，值是各行业信息
    indusrty_dict = get_industry(securities, date=watch_date)

    # get()函数，返回 level 关键词对应的值，然后提取次级中 method 的值，如果没有值，返回 np.nan
    # 如从 industry_dict 中查找每只股票的 level=sw_l1 的信息，返回其 method=industry_code 的值，如 '银行I'
    indusrty_ser = pd.Series({k: v.get(level, {method: np.nan})[method] for k, v in indusrty_dict.items()})
    indusrty_ser.name = method.upper()
    return indusrty_ser


# 获取股价
def get_stock_price(security,
                    periods
                    ):
    """
    获取对应频率的收盘价

    Parameters
    -----------
    security: List
        股票代码
    periods: List
        时间区间

    Returns
    ------------
    yield: pd.DataFrame

    """
    for trade in tqdm_notebook(periods, desc='获取收盘价数据'):
        yield get_price(security,
                        end_date=trade,
                        count=1,
                        fields='close',
                        fq='post',  # 后复权
                        panel=False)


# 获取股票池的区间收盘价数据
def get_pool_period_price(securities,
                          periods
                          ):
    """
    获取股票池的区间收盘价数据

    Parameters
    ----------
    securities : List
        股票代码
    periods : List
        时间区间

    Returns
    -------
     : pd.DataFrame

    """
    prices_list = list(get_stock_price(securities, periods))
    price_df = pd.concat(prices_list)
    return price_df


# 获取市值表数据
def get_factor_valuation(securities,
                         periods,
                         fields,
                         desc
                         ):
    """
    获取多个标的在指定交易日范围内的市值表数据

    Parameters
    ----------
    securities : List
        股票代码
    periods : List
        时间区间
    fields : List
        默认获取PB数据
    desc : str

    Returns
    -------
    yield: pd.DataFrame

    """
    if fields is None:
        fields = ['pb_ratio']

    for trade in tqdm_notebook(periods, desc=desc):
        yield get_valuation(securities,
                            end_date=trade,
                            fields=fields,
                            count=1)


# 获取股票池
class StocksPool(object):
    """
    这是一个可通用的类，用于获取某日的成分股股票
    1. 过滤st
    2. 过滤上市不足N个月
    3. 过滤当月交易不超过N日的股票
    ---------------
    输入参数：
        index_symbol:指数代码,A等于全市场,
        watch_date:日期
    """
    def __init__(self, symbol, watch_date):
        """

        Parameters
        ----------
        symbol : str
        watch_date : dt.date or str
        """
        if isinstance(watch_date, str):
            self.watch_date = pd.to_datetime(watch_date).date()
        else:
            self.watch_date = watch_date
        self.symbol = symbol
        self.get_index_component_stocks()

    # 获取指数成分股，建立基础股票池
    def get_index_component_stocks(self):
        if self.symbol == 'A':
            # 返回当日所有的股票列表，但不判断是否可交易
            wd: pd.DataFrame = get_all_securities(types=['stock'], date=self.watch_date)
            self.securities: List = wd.query('end_date != "2200-01-01"').index.tolist()
        else:
            # 取该指数的成分股
            self.securities: List = get_index_stocks(self.symbol, self.watch_date)

    # 过滤停牌股
    def filter_paused(self, paused_N=1, threshold=None):
        """

        Parameters
        ----------
        paused_N : int
            默认为1即查询当日不停牌
        threshold : int
            在过paused_N日内停牌数量小于threshold，如果希望区间内完全没有停牌股票， paused_N 应等于 threshold

        Returns
        -------

        """

        """
        -----
        输入:
            paused_N:
            threshold:
            
        """
        if (threshold is not None) and (threshold > paused_N):
            raise ValueError(f"参数threshold天数不能大于paused_N天数")

        # 将基于查询时点 watch_date 向前推 paused_N 个交易日的数据，取停牌字段，重组为透视表，停牌为1
        paused = get_price(self.securities, end_date=self.watch_date, count=paused_N, fields='paused', panel=False)
        paused = paused.pivot(index='time', columns='code')['paused']

        # 如果threhold不为None 获取过去paused_N内停牌数少于threshodl天数的股票
        if threshold:
            # 求和后，sum_paused_day 的值为时间区间内停牌的次数，如停牌2次，则该值为2
            sum_paused_day = paused.sum()
            # 取出所有停牌次数少于 threshold 的股票，存为列表
            self.securities = sum_paused_day[sum_paused_day < threshold].index.tolist()

        else:
            # 如果不指定阈值，则返回最新一日未停牌的股票列表
            paused_ser = paused.iloc[-1]
            self.securities = paused_ser[paused_ser == 0].index.tolist()

    # 过滤ST
    def filter_st(self):
        # get_extras API 可直接返回当日是否为 ST 的结果，是为 True
        extras_ser = get_extras('is_st', self.securities, end_date=self.watch_date, count=1).iloc[-1]
        # 返回当日不是 ST 的列表
        self.securities = extras_ser[extras_ser == False].index.tolist()

    # 过滤上市天数不足以threshold天的股票
    def filter_ipodate(self, threshold=180):
        """

        Parameters
        ----------
        threshold : int
            默认为180日

        Returns
        -------

        """
        def _check_ipodate(code: str, watch_date: dt.date) -> bool:
            # 检查上市至今的日期是否大于阈值，是为 True
            code_info = get_security_info(code)

            if (code_info is not None) and ((watch_date - code_info.start_date).days > threshold):

                return True

            else:

                return False

        # 如果大于阈值则纳入代码表
        self.securities = [code for code in self.securities if _check_ipodate(code, self.watch_date)]

    # 过滤行业
    def filter_industry(self, industry=None, level='sw_l1', method='industry_name'):
        """

        Parameters
        ----------
        industry : List
        level : str
        method : str
        """
        ind = get_stock_ind(self.securities, self.watch_date, level, method)
        target = ind.to_frame('industry').query('industry != @industry')
        self.securities = target.index.tolist()


"""
============================================================================================================================================================
                                                                     指标和绩效计算
============================================================================================================================================================
"""


def get_return(price_df):
    """
    计算收益率

    Parameters
    ----------
    price_df : pd.DataFrame
        收盘价数据

    Returns
    -------
    return_df : pd.DataFrame

    """
    pivot_price = pd.pivot_table(price_df, index='time', columns='code', values='close')
    return_df = pivot_price.pct_change().shift(-1)
    return_df = return_df.iloc[:-1].stack()
    return return_df


def get_benchmark_return(benchmark='000300.XSHG',
                         period=None):
    """
    获取基准标的的收益率

    Parameters
    ----------
    benchmark : str
         默认为沪深300
    period : List

    Returns
    -------
    benchmark_ret : pd.DataFrame

    """
    benchmark = list(get_factor_price(benchmark, period))
    benchmark = pd.concat(benchmark)
    benchmark_ret = benchmark['close'].pct_change().shift(-1)
    return benchmark_ret


def strategy_performance(return_df,
                         periods='daily',
                         benchmark=None
                         ):
    """
    计算风险指标 默认为年化数据

    Parameters
    ----------
    return_df : pd.DataFrame
        策略的日回报，非累积
    periods : str
        定义收益数据的周期性以计算年化，{‘monthly’:12, ’weekly’:52, ’daily’:252}
    benchmark : str
        基准收益率

    Returns
    -------
    ser : pd.DataFrame

    """
    ser: pd.DataFrame = pd.DataFrame()
    ser['年化收益率'] = ep.annual_return(return_df, period=periods)
    ser['累计收益'] = ep.cum_returns(return_df).iloc[-1]

    ser['夏普比'] = ep.sharpe_ratio(return_df, risk_free=0, period=periods, annualization=None)
    ser['索提诺'] = ep.sortino_ratio(return_df, required_return=0, period=periods, annualization=None, _downside_risk=None)
    ser['卡玛比率'] = ep.calmar_ratio(return_df, period=periods, annualization=None)
    ser['欧米加比率'] = ep.omega_ratio(return_df, risk_free=0.0, required_return=0.0, annualization=252)

    ser['波动率'] = return_df.apply(lambda x: ep.annual_volatility(x, period=periods))
    # ser['最大回撤'] = return_df.apply(lambda x: ep.max_drawdown(x))
    ser['最大回撤'] = ep.max_drawdown(return_df)

    if benchmark is not None:
        select_col = [col for col in return_df.columns if col != benchmark]
        ser['IR'] = return_df[select_col].apply(lambda x: information_ratio(x, return_df[benchmark]))
        ser['Alpha'] = return_df[select_col].apply(lambda x: ep.alpha(x, return_df[benchmark], period=periods))
        ser['超额收益'] = ser['年化收益率'] - ser.loc[benchmark, '年化收益率']  # 计算相对年化波动率

    ser = ser.T
    return ser


def information_ratio(returns,
                      factor_returns):
    """
    计算策略的 IR

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        策略的日回报，参考格式 empyrical.stats.cum_returns
    factor_returns: pd.Series or pd.DataFrame
        基准收益率

    Returns
    -------
    ir : float
        IR 值
    """

    def _adjust_returns(returns,
                        adjustment_factor):
        """
        解决收益率为0的情况，修正收益率

        Parameters
        ----------
        returns : pd.Series
        adjustment_factor : pd.Series or float

        Returns
        -------
        rt : pd.Series
        """
        if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
            return returns.copy()
        return returns - adjustment_factor

    if len(returns) < 2:
        return np.nan

    active_return = _adjust_returns(returns, factor_returns)
    tracking_error = np.std(active_return, ddof=1)
    if np.isnan(tracking_error):
        return 0.0
    if tracking_error == 0:
        return np.nan
    ir = np.mean(active_return) / tracking_error
    return ir


"""
============================================================================================================================================================
                                                                        因子合成
============================================================================================================================================================
"""


# 标记得分
def sign(ser):
    """
    标记分数,正数为1,负数为0

    Parameters
    ----------
    ser : pd.Series

    Returns
    -------
    ser : pd.Series

    """
    return ser.apply(lambda x: np.where(x > 0, 1, 0))


# 因子构造器
class ScoreFactor(Factor):
    name = 'FScore'
    # 获取数据的最长时间窗口，返回的是日级别的数据，需要使用 pd.DataFrame 转换
    max_window = 1
    watch_date = None
    # paidin_capital 实收资本 股本变化会反应在该科目中
    # 依赖的基础因子名称 https://www.joinquant.com/help/api/help#factor:%E5%9B%A0%E5%AD%90%E5%AE%9A%E4%B9%89
    dependencies = [
        'roa', 'roa_4', 'net_operate_cash_flow', 'total_assets',
        'total_assets_1', 'total_assets_4', 'total_assets_5',
        'operating_revenue', 'operating_revenue_4',
        'total_non_current_assets', 'total_non_current_liability',
        'total_non_current_assets_4', 'total_non_current_liability_4',
        'total_current_assets', 'total_current_liability',
        'total_current_assets_4', 'total_current_liability_4',
        'gross_profit_margin', 'gross_profit_margin_4', 'paidin_capital',
        'paidin_capital_4'
    ]
    #
    def calc(self, data: Dict) -> None:
        roa: pd.DataFrame = data['roa']

        cfo: pd.DataFrame = data['net_operate_cash_flow'] / data['total_assets']

        delta_roa: pd.DataFrame = roa / data['roa_4'] - 1

        accrual: pd.DataFrame = cfo - roa * 0.01

        # 杠杆变化
        ## 变化为负数时为1，否则为0 取相反
        leveler: pd.DataFrame = data['total_non_current_liability'] / data['total_non_current_assets']
        leveler1: pd.DataFrame = data['total_non_current_liability_4'] / data['total_non_current_assets_4']
        delta_leveler: pd.DataFrame = -(leveler / leveler1 - 1)

        # 流动性变化
        liquid: pd.DataFrame = data['total_current_assets'] / data['total_current_liability']
        liquid_1: pd.DataFrame = data['total_current_assets_4'] / data['total_current_liability_4']
        delta_liquid: pd.DataFrame = liquid / liquid_1 - 1

        # 毛利率变化
        delta_margin: pd.DataFrame = data['gross_profit_margin'] / data['gross_profit_margin_4'] - 1

        # 是否发行普通股权
        eq_offser: pd.DataFrame = data['paidin_capital'] / data['paidin_capital_4'] - 1

        # 总资产周转率
        total_asset_turnover_rate: pd.DataFrame = data['operating_revenue'] / (data['total_assets'] + data['total_assets_1']).mean()

        total_asset_turnover_rate_1: pd.DataFrame = data['operating_revenue_4'] / (data['total_assets_4'] + data['total_assets_5']).mean()

        # 总资产周转率同比
        delta_turn: pd.DataFrame = total_asset_turnover_rate / total_asset_turnover_rate_1 - 1

        indicator_tuple: Tuple = (roa, cfo, delta_roa, accrual, delta_leveler, delta_liquid, delta_margin, delta_turn, eq_offser)

        # 储存计算FFscore所需原始数据
        self.basic: pd.DataFrame = pd.concat(indicator_tuple).T.replace([-np.inf, np.inf], np.nan)
        #
        self.basic.columns = ['ROA', 'CFO', 'DELTA_ROA', 'ACCRUAL', 'DELTA_LEVELER', 'DLTA_LIQUID', 'DELTA_MARGIN', 'DELTA_TURN', 'EQ_OFFSER']
        self.fscore: pd.Series = self.basic.apply(sign).sum(axis=1)


# 因子获取
def get_factor(symbol,
               factor,
               periods,
               filter_industry=None
               ):
    """
    获取因子得分

    Parameters
    ---------
    symbol : str
        输入A表示全A股票池或者输入指数代码
    factor : Factor
        不同的fscore模型
    periods : List
        计算得分的时间序列
    filter_industry : List or str
        传入需要过滤的行业

    Return
    ----------
    yield : pd.Series or pd.DataFrame
        最终得分, 财务数据
    """
    for trade in tqdm_notebook(periods, desc=str(factor) + '因子获取'):
        # 获取股票池
        stock_pool_func = StocksPool(symbol, trade)
        stock_pool_func.filter_paused(22, 21)  # 过滤22日停牌超过21日的股票
        stock_pool_func.filter_st()  # 过滤st
        stock_pool_func.filter_ipodate(180)  # 过滤次新
        if filter_industry:  # 是否过滤行业
            stock_pool_func.filter_industry(filter_industry)

        my_factor = factor()
        my_factor.watch_date = trade

        # 因子计算
        # 返回一个 dict 对象, key 是各 factors 的 name，value 是一个DataFrame，index 是日期， column 是股票代码
        calc_factors(stock_pool_func.securities, [my_factor], start_date=trade, end_date=trade)
        # 使用生成器，返回因子值
        yield my_factor


def get_period_factor(symbol,
                      factor,
                      periods,
                      filter_industry=None
                      ):
    """
    获取FFScore得分，每个都是 yield 对象

    Parameters
    ----------
    symbol : str
    factor : Factor
    periods : List
    filter_industry : List or str

    Returns
    -------

    """
    data_list = list(get_factor(symbol, factor, periods, filter_industry))
    # 因子基础值
    factor_basic_df: pd.DataFrame = pd.concat({pd.to_datetime(f.watch_date): f.basic for f in data_list}, names=['date', 'asset'])
    # 因子值
    factor_df: pd.DataFrame = pd.concat({pd.to_datetime(f.watch_date): f.score for f in data_list}, names=['date', 'asset'])
    return factor_basic_df, factor_df


"""
============================================================================================================================================================
                                                                        通用方法
============================================================================================================================================================
"""


# 获取年末季末时点
def get_trade_period(start_date,
                     end_date,
                     freq='ME'
                     ):
    """
    获取年末季末时点

    Parameters
    ----------
    start_date : str
    end_date : str
        start_date/end_date:str YYYY-MM-DD
    freq : str
        M月，Q季,Y年 默认ME
        E代表期末 S代表期初

    Returns
    -------
    date : List[dt.date]
    """
    days = pd.Index(pd.to_datetime(get_trade_days(start_date, end_date)))
    idx_df = days.to_frame()

    if freq[-1] == 'E':
        day_range = idx_df.resample(freq[0]).last()
    else:
        day_range = idx_df.resample(freq[0]).first()

    day_range = day_range[0].dt.date

    return day_range.dropna().values.tolist()


# 查询当前交易日向前 count 个交易日的日期对象
def get_before_after_trade_days(date,
                                count,
                                is_before=True):
    """
    查询当前交易日向前 count 个交易日的日期对象

    Parameters
    ----------
    date :
        查询日期
    count :
        前后追朔的数量
    is_before :
        True , 前count个交易日  ; False ,后count个交易日

    Returns
    -------
    all_date : List[dt.date]
        基于date的日期, 向前或者向后count个交易日的日期 ,一个datetime.date 对象

    """
    all_date = pd.Series(get_all_trade_days())

    if isinstance(date, str):
        date = dt.datetime.strptime(date, '%Y-%m-%d').date()
    if isinstance(date, dt.datetime):
        date = date.date()

    if is_before:
        return all_date[all_date <= date].tail(count).values[0]
    else:
        return all_date[all_date >= date].head(count).values[-1]


# 数据存储
def pkl_batch(file,
              file_name,
              method
              ):
    """
    将数据保存为 pkl 格式，或者读取

    Parameters
    ----------
    file : str
        用于存储的数据
    file_name : str
        pkl 文件名
    method : str
        传入是写入还是读取

    Returns
    -------

    """
    if method == 'write':
        pickle_file = open(file_name + '.pkl', 'wb')
        pickle.dump(file, pickle_file)
        pickle_file.close()
        return True
    elif method == 'read':
        pickle_file = open(file_name + '.pkl', 'rb')
        data = pickle.load(pickle_file)
        pickle_file.close()
        return data
    else:
        return False


"""
============================================================================================================================================================
                                                                        绘图函数
============================================================================================================================================================
"""
# 配置全局变量
color_map = ['red', 'green', 'cyan', 'magenta', 'blue']


# 绘制各分组的累计收益统计图
def plot_cum_return(title,
                    ret,
                    benchmark_ret,
                    benchmark_name
                    ):
    """
    绘制各分组的累计收益图

    Parameters
    ----------
    title : str
        图表标题
    ret : pd.DataFrame
        分组的收益率 df 格式文件
    benchmark_ret : pd.DataFrame
        基准的收益率 df 格式文件
    benchmark_name : str
        基准名称

    Returns
    -------
    plot : plt
        绘图

    """
    fig, ax = plt.subplots(figsize=(26, 6))
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: '%.2f%%' % (x * 100)))
    ep.cum_returns(ret).plot(ax=ax, color=color_map)
    if not benchmark_ret.empty:
        ep.cum_returns(benchmark_ret.reindex(ret.index)).plot(ax=ax, label=benchmark_name, color='darkgray', ls='--')
    ax.axhline(0, color='black', lw=1)
    plt.legend()


# 绘制相关性的热力图
def plot_heatmap(df,
                 title
                 ):
    """
    绘制相关性的热力图

    Parameters
    ----------
    df :  pd.DataFrame
        需要绘制的 df 格式文件
    title : str
        标题

    Returns
    -------
    plt : plt
        绘图

    """
    fig, ax = plt.subplots(figsize=(26, 6))
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    ax.set_title(title)
    sns.heatmap(corr, annot=True, linewidths=.5, mask=mask)


__all__ = [
    # 封装的 jqdata 数据处理
    'get_stock_ind',  'get_stock_price', 'get_pool_period_price', 'get_factor_valuation',

    # 指标和绩效计算
    'get_return', 'get_benchmark_return', 'strategy_performance', 'information_ratio',

    # 因子合成
    'get_factor', 'get_period_factor',

    # 通用方法
    'get_trade_period', 'get_before_after_trade_days', 'pkl_batch',

    # 绘图函数
    'plot_cum_return', 'plot_heatmap'
]
