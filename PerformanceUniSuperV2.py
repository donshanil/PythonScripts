# This is a sample Python script.
import pandas as pd
import sqlite3 as sql
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'


def connect(path):
    """Connects to sqlite db and returns connection object"""
    return sql.connect(path)


def get_data(connection, tables=None):
    """ Returns a list of dataframes - if no parameter is given returns dateframe with default table names"""
    if tables is None:
        return [table_to_pd(connection, Tables['Holdings_data']),
                table_to_pd(connection, Tables['Portfolio_data']),
                table_to_pd(connection, Tables['Pricing_data']),
                table_to_pd(connection, Tables['Security_data'])]
    # Put logic for default tables here...maybe write with variadic arguments to allow for general solution?


def get_date_range(start_date, end_date, df, col=None):
    """Returns data frame where filter is applied over given data range. Default column name is date.
    Put logic in if not."""
    if col is None:
        return df.loc[(df.date >= start_date & df.date <= end_date)]
    # put logic for non default here


def get_portfolio_pricing(pricing_data, portfolio_names=None):
    """Helper function - separates aggregated data set into portfolios by name. Default logic set."""
    if portfolio_names is None:
        return [pricing_data.loc[(pricing_data.portfolio_id == 1)],
                pricing_data.loc[(pricing_data.portfolio_id == 2)],
                pricing_data.loc[(pricing_data.portfolio_id == 3)]]
    # For non-default logic set here


def get_returns(start_date, end_date, df, cols=None):
    """ Helper function that returns returns pct returns over a given period"""
    if cols is None:
        cols = ['date', 'price']
    df_date = get_date_range(start_date, end_date)
    df_date.sort_values(cols[0], inplace=True)
    df_date['price_return'] = df.cols[1].pct_change()
    return df_date


def get_portfolio_details(port_names, excess_ret, vols, sharp_ratios):
    my_port = {'Portfolio Name': port_names, 'Excess Returns': excess_ret, 'Volatility': vols,
               'Sharp Ratio': sharp_ratios}
    return pd.DataFrame.from_dict(my_port)


def total_mkt_value(df, column_name):
    return df[column_name].sum()


def table_to_pd(connection, tablename):
    """Get table from db - needs a connection object and tablename"""
    sql_query = "SELECT * FROM " + tablename
    return pd.read_sql_query(sql_query, connection)


def get_security_pricing(pricing_table, security_table):
    """Returns a dataframe sorted by date joined along the security_id"""
    df = pd.merge(pricing_table, security_table, how='left', left_on='security_id', right_index=True)[
        ['date', 'security_id', 'code', 'price', 'market_cap']]
    df.sort_values(by='date')
    return df


def get_portfolio_securities_list(portfolio):
    """helper function - returns security list if passed in a portfolio.
    nb - is not a unique list and is not sorted"""
    return portfolio['code'].tolist()


def get_valuation_dates(portfolio):
    """helper function -- returns unique dates list for portfolio. Expects security price table
    nb - write this better in v2. results are unique but not sorted!"""
    return portfolio['date'].unique().sort_values('date').tolist()


def get_portfolio_weights(portfolio_df):
    """Returns a dataframe after calculating weights of portfolio"""
    df = portfolio_df[['date', 'portfolio_id', 'security_id', 'code', 'holding', 'price']]
    df['total_position'] = df.holding * df.price
    date_grp_total = df.groupby('date')[['total_position']].sum().reset_index()
    date_grp_asset = df.groupby(['date', 'security_id', 'code']).apply(
        lambda x: x['holding'] * x['price']).reset_index()
    df_1 = pd.merge(date_grp_asset, date_grp_total, on=['date'])
    df_1['weight'] = df_1[0] / df_1['total_position']
    return df_1[['date', 'security_id', 'code', 'total_position', 'weight']]


def get_portfolio_weighted_returns(portfolio):
    # Potentially easier to do this by grouping across security and date together.
    # Variable holds the dataframe containg the security level returns of the portfolio (price over price)
    Portfolio_return = pd.DataFrame(columns=['date', 'code', 'price_return', 'weighted_return'])
    # group here by security code (we the returns for each period for each security)
    g = portfolio.groupby('code')
    # loop through all the groups, for each group dataframe apply the pct change func to the price column
    # The dates need to be ordered correctly here - this should be checked in a prod version/or done here
    for security, dataframe in g:
        dataframe['price'] = dataframe.price.pct_change()[1:]
        dataframe.rename(columns={'price': 'price_return'}, inplace=True)
        dataframe['weighted_return'] = dataframe['price_return'] * dataframe['weight'] * 100
        dataframe.dropna(inplace=True)
        Portfolio_return = pd.concat([Portfolio_return, dataframe])
    total = Portfolio_return.groupby(['date', 'portfolio_id'])[['weighted_return']].sum()
    return total


def plot_all_portfolios(portfolio_list):
    ax = portfolio_list[0].reset_index().plot(x='date', y='weighted_return', label='Portfolio1')
    portfolio_list[1].reset_index().plot(ax=ax, x='date', y='weighted_return', label='Portfolio2')
    portfolio_list[2].reset_index().plot(ax=ax, x='date', y='weighted_return', label='Portfolio3')
    plt.title('Portfolio Weighted Returns')
    plt.xlabel('Dates')
    plt.ylabel('Pct Weighted Returns')
    plt.show()


def plot_portfolios(portfolio_list, x_value=None, y_value=None):
    if (x_value is None) and (y_value is None):
        for each_portfolio in portfolio_list:
            each_portfolio.reset_index().plot(x='date', y='weighted_return')
            plt.show()
    else:
        for each_portfolio in portfolio_list:
            each_portfolio.reset_index()(x=x_value, y=y_value)


def rebalance(securities_no, rebal_date, pricing_data):
    index_data = pricing_data[pricing_data['date'] == rebal_date]
    top_n_index = index_data.nlargest(securities_no, ['market_cap'])
    top_n_index['total_index_value'] = top_n_index['market_cap'].sum()
    top_n_index['weight'] = (top_n_index['market_cap'] / top_n_index['total_index_value'])
    return top_n_index


def add_quaterly_rebalance(securities_no, data, initial_index):
    rebal_dates = ['20170331', '20170630', '20170929', '20171229', '20180329', '20180629', '20180928', '20181228',
                   '20190329', '20190628', '20190927', '20191227', '20200327', '20200626', '2020925',
                   '20201231', '20210326', '20210625', '20210924', '20211231', '20220325', '20220624']
    df1 = initial_index
    for each_date in rebal_dates:
        df2 = rebalance(securities_no, each_date, data)
        df1 = pd.concat([df1, df2])
        df2.drop(df2.index, inplace=True)
    return df1


def str_to_date(s):
    return pd.to_datetime(s)


def convert_todate(df, column):
    return df[column].map(lambda s: str_to_date(s))


def get_index_data(index, date):
    return index[index['date'] == date][['security_id', 'price', 'weight']]


def get_excess_return(portfolio, index):
    port = portfolio.reset_index()
    idx = index.reset_index()
    idx_dates = idx['date'].tolist()
    port['cum_return'] = port['weighted_return'].cumsum()
    # Get subset of dates where index and portfolio match - read off cum return
    port = port[port['date'].isin(idx_dates)]
    port_vs_idx = pd.merge(port, idx, left_on='date', right_on='date', how='right')
    port_vs_idx['difference'] = port_vs_idx['weighted_return_y'] - port_vs_idx['cum_return']
    port_vs_idx = port_vs_idx[['date', 'difference']]
    return port_vs_idx


def get_portfolio_excess_returns(port, index, start_date=None, end_date=None):
    if (start_date is None) and (end_date is None):
        # This makes it simpler than accessing with index
        index = index.reset_index()
        port = port.reset_index()
        index.set_index('date', inplace=True)
        port.set_index('date', inplace=True)
        i_0 = index['weighted_return'][EXCESS_RETURN_START]
        i_t = index['weighted_return'][EXCESS_RETURN_END]
        index_return = (i_t / i_0 - 1)
        p_0 = port['weighted_return'][EXCESS_RETURN_START]
        p_t = port['weighted_return'][EXCESS_RETURN_END]
        port_return = (p_t / p_0 - 1)
        return port_return - index_return


def get_portfolio_std(portfolio, columnname='weighted_return'):
    df = portfolio.reset_index()
    return df.apply(lambda x: df.loc[
        (df['date'] >= EXCESS_RETURN_START) & (df['date'] <= EXCESS_RETURN_END), 'weighted_return'].std(),
                    axis=1).max()  # portfolio[columnname].std()


def portfolio_excess_return(excess_ret_df, startdate, enddate):
    df = excess_ret_df
    return df.apply(lambda x: df.loc[(df.date >= startdate) & (df.date <= enddate), 'difference'].sum(),
                    axis=1).max()


def get_sharp_ratio(expected_return, risk_free_rate, volatility):
    return (expected_return - risk_free_rate) / volatility


def get_portfolio_expected_returns(port, start_date, end_date, col=None):
    if col is None:
        port = port.reset_index()
        port.set_index('date', inplace=True)
        port = port.loc[start_date:end_date]
        return port['weighted_return'].mean()


def get_total_position_value(df, date, column_name):
    return df[df['date'] == date].iloc[0, df.columns.get_loc(column_name)]


def get_excess_return_new(port, index, column_names, start_date=None, end_date=None):
    if start_date is None and end_date is None:
        p0 = get_total_position_value(port, EXCESS_RETURN_START, column_names[0])
        p1 = get_total_position_value(port, EXCESS_RETURN_END, column_names[0])
        i0 = get_total_position_value(index, EXCESS_RETURN_START, column_names[1])
        i1 = get_total_position_value(index, EXCESS_RETURN_END, column_names[1])
        port_return = p1 / p0 - 1
        index_return = i1 / i0 - 1
        if (port_return > 0 and index_return > 0) or (port_return < 0 and index_return < 0):
            return port_return - index_return
        elif port_return < 0 and index_return > 0:
            return -1 * (abs(port_return) + index_return)
        elif port_return > 0 and index_return < 0:
            return 1 * (port_return + abs(index_return))


if __name__ == '__main__':
    # Filepath
    db_path = r'C:\Users\U3BR\Downloads\unisuper_assessment.db'
    # Index definitions
    TOP30 = 30
    START_DATE = '20161230'
    EXCESS_RETURN_START = '20210625'
    EXCESS_RETURN_END = '20220624'
    # Yield on AUD government bond curve at the 6 year mark
    RISK_FREE_RATE = 2.969 / 100

    # Tables
    Tables = {'Holdings_data': 'HOLDING', 'Portfolio_data': 'PORTFOLIO',
              'Pricing_data': 'PRICE', 'Security_data': 'SECURITY'}
    myPortfolios = {'Portfolio1': 1, 'Portfolio2': 2, 'Portfolio3': 3}

    db_conn = connect(db_path)

    # Dataframes
    Holdings_df, Portfolio_df, Pricing_df, Security_df = get_data(db_conn)
    # Add/Merge security code to pricing data
    Security_pricing = get_security_pricing(Pricing_df, Security_df)

    # Merge Portfolio data and security pricing data
    Portfolio_pricing = pd.merge(Holdings_df, Security_pricing, how='left', left_on=['security_id', 'date'],
                                 right_on=['security_id', 'date']).sort_values(by='date')
    # Separate data at portfolio level
    Portfolios = get_portfolio_pricing(Portfolio_pricing)

    Weights = [get_portfolio_weights(Portfolios[0]),
               get_portfolio_weights(Portfolios[1]), get_portfolio_weights(Portfolios[2])]

    # Merge Weights to portfolio level data set
    Weighted_Portfolios = [
        pd.merge(Portfolios[0], Weights[0], how='left', left_on=['date', 'code'], right_on=['date', 'code']),
        pd.merge(Portfolios[1], Weights[1], how='left', left_on=['date', 'code'], right_on=['date', 'code']),
        pd.merge(Portfolios[2], Weights[2], how='left', left_on=['date', 'code'], right_on=['date', 'code'])]

    Weighted_returns = [get_portfolio_weighted_returns(Weighted_Portfolios[0]),
                        get_portfolio_weighted_returns(Weighted_Portfolios[1]),
                        get_portfolio_weighted_returns(Weighted_Portfolios[2])]

    # Plotting portfolio returns
    plot_all_portfolios(Weighted_returns)

    # Index calculations
    # Create index top 30, market capped. Calculate rebalanced for quarter ends
    top30_index = rebalance(TOP30, START_DATE, Pricing_df)
    top30_index_rebalanced = add_quaterly_rebalance(TOP30, Pricing_df, top30_index).sort_values(by='date')
    port_1r, port_2r, port_3r = (
        get_excess_return_new(Weights[0], top30_index_rebalanced, ['total_position', 'total_index_value']),
        get_excess_return_new(Weights[1], top30_index_rebalanced, ['total_position', 'total_index_value']),
        get_excess_return_new(Weights[2], top30_index_rebalanced, ['total_position', 'total_index_value']),)
    top30_index_rebalanced = pd.merge(top30_index_rebalanced, Security_pricing.drop_duplicates(subset=['security_id']),
                                      how='left', on=['security_id'])

    # Want to re-use portfolio returns function used previously, so we setup index dataframe with same format.
    # Get relevant columns needed.
    top30_index_rebalanced = top30_index_rebalanced[['security_id', 'date_x', 'code', 'price_x', 'weight']].sort_values(
        by='date_x')
    top30_index_rebalanced.rename(columns={'date_x': 'date', 'price_x': 'price'}, inplace=True)
    # Add 'dummy' portfolio id as weighted returns function uses the portfolio id - maybe design that better
    top30_index_rebalanced['portfolio_id'] = 'Index30'

    # call weighted returns function on index portfolio.
    top30_quaterly_weighted_returns = get_portfolio_weighted_returns(top30_index_rebalanced)
    index_volatility = get_portfolio_std(top30_quaterly_weighted_returns)
    expected_index_return = get_portfolio_expected_returns(top30_quaterly_weighted_returns, EXCESS_RETURN_START,
                                                           EXCESS_RETURN_END)
    index_sharpe_ratio = get_sharp_ratio(expected_index_return, RISK_FREE_RATE, index_volatility)

    expected_returns = {
        'Portfolio1': get_portfolio_expected_returns(Weighted_returns[0], EXCESS_RETURN_START, EXCESS_RETURN_END),
        'Portfolio2': get_portfolio_expected_returns(Weighted_returns[1], EXCESS_RETURN_START, EXCESS_RETURN_END),
        'Portfolio3': get_portfolio_expected_returns(Weighted_returns[2], EXCESS_RETURN_START, EXCESS_RETURN_END)}

    volatility = {'Portfolio1': get_portfolio_std(Weighted_returns[0]),
                  'Portfolio2': get_portfolio_std(Weighted_returns[1]),
                  'Portfolio3': get_portfolio_std(Weighted_returns[2]),
                  'TOP30_index': index_volatility}

    excess_return_annual = {
        'Portfolio1': port_1r,
        'Portfolio2': port_2r,
        'Portfolio3': port_3r,
        'TOP30_index': 0}

    sharp_ratio = {
        'Portfolio1': get_sharp_ratio(expected_returns['Portfolio1'], RISK_FREE_RATE, volatility['Portfolio1']),
        'Portfolio2': get_sharp_ratio(expected_returns['Portfolio2'], RISK_FREE_RATE, volatility['Portfolio2']),
        'Portfolio3': get_sharp_ratio(expected_returns['Portfolio3'], RISK_FREE_RATE, volatility['Portfolio3']),
        'TOP30_index': index_sharpe_ratio}

    print(
        get_portfolio_details(['Portfolio1', 'Portfolio2', 'Portfolio3', 'TOP30_index'], excess_return_annual.values(),
                              volatility.values(), sharp_ratio.values()))

    # Risk Return Plots
    X1, X2, Y1, Y2 = volatility['Portfolio1'], volatility['Portfolio2'], expected_returns['Portfolio1'] ,expected_returns['Portfolio2']
    X3, X4, Y3, Y4 = volatility['Portfolio1'], volatility['Portfolio3'], expected_returns['Portfolio1'] ,expected_returns['Portfolio3']
    X5, X6, Y5, Y6 = volatility['Portfolio2'], volatility['Portfolio3'], expected_returns['Portfolio2'] ,expected_returns['Portfolio3']
    plt.xlabel('Risk(sigma)')
    plt.ylabel('Expected portfolio return E(R)')
    plt.title('Risk Return Plot')
    plt.scatter([X1, X2], [Y1, Y2], )
    plt.show()
    plt.scatter([X3, X4], [Y3, Y4])
    plt.show()
    plt.scatter([X5, X6], [Y5, Y6])
    plt.show()
