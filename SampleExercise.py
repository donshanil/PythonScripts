# Sample Python Scrip for assignment
import pandas as pd
import sqlite3 as sql
import matplotlib.pyplot as plt

# Turns off warning due to pandas not parsing chained assignments.
# Unsure if this is Pandas version specific.
pd.options.mode.chained_assignment = None  # default='warn'


def connect(path):
    """Connects to sqlite db and returns connection object"""
    return sql.connect(path)


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
    return portfolio['date'].unique().tolist()


def get_portfolio_weights(portfolio_df):
    """Returns a dataframe after calculating weights of portfolio"""
    df = portfolio_df[['date', 'security_id', 'code', 'holding', 'price']]
    df['total_position'] = df.holding * df.price
    total_portfolio_value = df['total_position'].sum()
    df['weight'] = (df['total_position'] / total_portfolio_value)
    return df[['date', 'security_id', 'code', 'weight']]


def get_portfolio_weighted_returns(portfolio):
    """Returns data frame with weighted returns for a give portfolio dataframe"""
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
    """Returns plot of portfolios -- requires portfolios as list"""
    ax = portfolio_list[0].reset_index().plot(x='date', y='weighted_return', label='Portfolio1')
    portfolio_list[1].reset_index().plot(ax=ax, x='date', y='weighted_return', label='Portfolio2')
    portfolio_list[2].reset_index().plot(ax=ax, x='date', y='weighted_return', label='Portfolio3')
    plt.title('Portfolio Weighted Returns')
    plt.xlabel('Dates')
    plt.ylabel('Pct Weighted Returns')
    plt.show()


def plot_portfolios(portfolio_list):
    """Creates separate plot for each portfolio in portfolio list -- requires portfolios as a list"""
    for each_portfolio in portfolio_list:
        each_portfolio.reset_index().plot(x='date', y='weighted_return')
        plt.show()


def rebalance(securities_no, rebal_date, pricing_data):
    """Returns top n index given n, rebalance date and pricing data (weights based off market cap)"""
    index_data = pricing_data[pricing_data['date'] == rebal_date]
    top_n_index = index_data.nlargest(securities_no, ['market_cap'])
    top_n_index['total_index_value'] = top_n_index['market_cap'].sum()
    top_n_index['weight'] = (top_n_index['market_cap'] / top_n_index['total_index_value'])
    return top_n_index


def add_quaterly_rebalance(securities_no, data, initial_index):
    """Returns dataframe of top 30 index for each rebalance date -- calls rebalance()"""
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
    """Helper function to convert string to datetime"""
    return pd.to_datetime(s)


def convert_todate(df, column):
    """Helper function to convert dataframe column to datetime"""
    return df[column].map(lambda s: str_to_date(s))


def get_index_data(index, date):
    """Helper function to return index data"""
    return index[index['date'] == date][['security_id', 'price', 'weight']]


def get_excess_return(portfolio, index):
    """Returns dataframe consisting of difference between portfolio and benchmark returns for each date in bench"""
    port = portfolio.reset_index()
    idx = index.reset_index()
    idx_dates = idx['date'].tolist()
    port['cum_return'] = port['weighted_return'].cumsum()
    port = port[port['date'].isin(idx_dates)]
    port_vs_idx = pd.merge(port, idx, left_on='date', right_on='date', how='right')
    port_vs_idx['difference'] = port_vs_idx['weighted_return_y'] - port_vs_idx['cum_return']
    port_vs_idx = port_vs_idx[['date', 'difference']]
    return port_vs_idx


def get_portfolio_std(portfolio):
    """Helper function - returns standard deviation of weighted returns"""
    df = portfolio.reset_index()
    return df.apply(lambda x: df.loc[
        (df['date'] >= EXCESS_RETURN_START) & (df['date'] <= EXCESS_RETURN_END), 'weighted_return'].std(), axis=1).max()


def get_portfolio_mean(portfolio, start_date, end_date):
    """Helper function -- returns mean of dataframe containing weighted returns for a given portfolio and date range"""
    df = portfolio.reset_index()
    return df.apply(lambda x: df.loc[(df.date >= start_date) & (df.date <= end_date), 'weighted_return'].mean(),
                    axis=1).max()


def get_portfolio_excessreturn(excess_ret_df, start_date, end_date):
    """Returns the excess return over a given date range"""
    df = excess_ret_df
    return df.apply(lambda x: df.loc[(df.date >= start_date) & (df.date <= end_date), 'difference'].sum(), axis=1).max()


def get_sharp_ratio(excess_return_difference, volatility_data):
    """Returns sharpe ratio"""
    return (excess_return_difference / volatility_data) * 1 / 100


def get_expected_excess_return(portfolio_mean, risk_free_rate):
    """Returns difference between expectation of excess returns given risk free rate"""
    return (portfolio_mean - risk_free_rate)


if __name__ == '__main__':
    # Filepath
    db_path = r'C:\Users\U3BR\Downloads\unisuper_assessment.db'
    # Index definitions
    TOP30 = 30
    START_DATE = '20161230'
    EXCESS_RETURN_START = '20210625'
    EXCESS_RETURN_END = '20220624'

    # Tables
    Tables = {'Holdings_data': 'HOLDING', 'Portfolio_data': 'PORTFOLIO', 'Pricing_data': 'PRICE',
              'Security_data': 'SECURITY'}
    myPortfolios = {'Portfolio1': 1, 'Portfolio2': 2, 'Portfolio3': 3}

    db_conn = connect(db_path)

    # Dataframes
    Holdings_df = table_to_pd(db_conn, Tables['Holdings_data'])
    Portfolio_df = table_to_pd(db_conn, Tables['Portfolio_data'])
    Pricing_df = table_to_pd(db_conn, Tables['Pricing_data'])
    Security_df = table_to_pd(db_conn, Tables['Security_data'])
    # Add/Merge security code to pricing data
    Security_pricing = get_security_pricing(Pricing_df, Security_df)
    # Merge Portfolio data and security pricing data
    Portfolio_pricing = pd.merge(Holdings_df, Security_pricing, how='left', left_on=['security_id', 'date'],
                                 right_on=['security_id', 'date']).sort_values(by='date')
    Portfolio_pricing.set_index(['date', 'portfolio_id', 'security_id'])

    # Separate data at portfolio level
    Portfolios = [Portfolio_pricing[Portfolio_pricing.portfolio_id == myPortfolios['Portfolio1']],
                  Portfolio_pricing[Portfolio_pricing.portfolio_id == myPortfolios['Portfolio2']],
                  Portfolio_pricing[Portfolio_pricing.portfolio_id == myPortfolios['Portfolio3']]]

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
    top30_index_rebalanced = pd.merge(top30_index_rebalanced, Security_pricing.drop_duplicates(subset=['security_id']),
                                      how='left', on=['security_id'])
    # Want to re-use portfolio returns function used previously, so we setup index dataframe with same format.
    # Get relevant columns needed.
    top30_index_rebalanced = top30_index_rebalanced[['security_id', 'date_x', 'code', 'price_x', 'weight']].sort_values(
        by='date_x')
    top30_index_rebalanced.rename(columns={'date_x': 'date', 'price_x': 'price'}, inplace=True)
    # Add 'dummy' portfolio id as weighted returns function uses the portfolio id - maybe design that better
    top30_index_rebalanced['portfolio_id'] = 'Index30'

    # call weighted returns function on index portfolio
    top30_quaterly_weighted_returns = get_portfolio_weighted_returns(top30_index_rebalanced)
    excess_return = {'Portfolio1': get_excess_return(Weighted_returns[0], top30_quaterly_weighted_returns),
                     'Portfolio2': get_excess_return(Weighted_returns[1], top30_quaterly_weighted_returns),
                     'Portfolio3': get_excess_return(Weighted_returns[2], top30_quaterly_weighted_returns)}

    volatility = {'Portfolio1': get_portfolio_std(Weighted_returns[0]),
                  'Portfolio2': get_portfolio_std(Weighted_returns[1]),
                  'Portfolio3': get_portfolio_std(Weighted_returns[2])}

    excess_return_annual = {
        'Portfolio1': get_portfolio_excessreturn(excess_return['Portfolio1'], EXCESS_RETURN_START, EXCESS_RETURN_END),
        'Portfolio2': get_portfolio_excessreturn(excess_return['Portfolio2'], EXCESS_RETURN_START, EXCESS_RETURN_END),
        'Portfolio3': get_portfolio_excessreturn(excess_return['Portfolio3'], EXCESS_RETURN_START, EXCESS_RETURN_END)}

    sharp_ratio = {'Portfolio1': get_sharp_ratio(excess_return_annual['Portfolio1'], volatility['Portfolio1']),
                   'Portfolio2': get_sharp_ratio(excess_return_annual['Portfolio2'], volatility['Portfolio2']),
                   'Portfolio3': get_sharp_ratio(excess_return_annual['Portfolio3'], volatility['Portfolio3'])}

    mean_return = {'Portfolio1': get_portfolio_mean(Weighted_returns[0], EXCESS_RETURN_START, EXCESS_RETURN_END),
                   'Portfolio2': get_portfolio_mean(Weighted_returns[1], EXCESS_RETURN_START, EXCESS_RETURN_END),
                   'Portfolio3': get_portfolio_mean(Weighted_returns[2], EXCESS_RETURN_START, EXCESS_RETURN_END)}

    # Output to screen
    excess_ret_str = "{0} 2021-2022 excess return is {1}%"
    volatility_str = "{0} 2021-2022 volatility is {1}"
    sharp_ratio_str = "{0} sharp ratio is {1}"
    print("***** Excess Returns *****")
    print(excess_ret_str.format('Portfolio1', round(excess_return_annual['Portfolio1'], 2)))
    print(excess_ret_str.format('Portfolio1', round(excess_return_annual['Portfolio2'], 2)))
    print(excess_ret_str.format('Portfolio1', round(excess_return_annual['Portfolio3'], 2)))
    print("***** Volatility *****")
    print(volatility_str.format('Portfolio1', round(volatility['Portfolio1'], 6)))
    print(volatility_str.format('Portfolio2', round(volatility['Portfolio2'], 4)))
    print(volatility_str.format('Portfolio3', round(volatility['Portfolio3'], 4)))
    print("***** Sharp Ratio *****")
    print(sharp_ratio_str.format('Portfolio1', round(sharp_ratio['Portfolio1'], 2)))
    print(sharp_ratio_str.format('Portfolio2', round(sharp_ratio['Portfolio2'], 2)))
    print(sharp_ratio_str.format('Portfolio3', round(sharp_ratio['Portfolio3'], 2)))
