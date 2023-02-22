import pandas as pd
from datetime import datetime
import sys
import os

wd = 'C:/Users/zacha/PycharmProjects/TennisPrediction'
os.chdir(wd)

def pull_atp(start_year, end_year):
    dfs = []
    for i in range(int(start_year), int(end_year)+1,1):
        print(f'Pulling {i} ATP match data')
        url = f'https://github.com/JeffSackmann/tennis_atp/blob/master/atp_matches_{i}.csv?raw=true'
        df = pd.read_csv(url, index_col=None,
                         header=0,
                         parse_dates=[5],
                         encoding = "ISO-8859-1",
                         date_parser = lambda t:pd.to_datetime(t))
        dfs.append(df)
    matches = pd.concat(dfs,ignore_index=True)
    return(matches)

def pull_futures(start_year, end_year):
    dfs = []
    for i in range(int(start_year), int(end_year)+1,1):
        print(f'Pulling {i} Futures match data')
        url = f'https://github.com/JeffSackmann/tennis_atp/blob/master/atp_matches_futures_{i}.csv?raw=true'
        df = pd.read_csv(url, index_col=None,
                         header=0,
                         parse_dates=[5],
                         encoding = "ISO-8859-1",
                         date_parser = lambda t:pd.to_datetime(t))
        dfs.append(df)
    matches = pd.concat(dfs,ignore_index=True)
    return(matches)

def pull_qual(start_year, end_year):
    dfs = []
    for i in range(int(start_year), int(end_year)+1,1):
        print(f'Pulling {i} Qualifier and Challenger match data')
        url = f'https://github.com/JeffSackmann/tennis_atp/blob/master/atp_matches_qual_chall_{i}.csv?raw=true'
        df = pd.read_csv(url, index_col=None,
                         header=0,
                         parse_dates=[5],
                         encoding = "ISO-8859-1",
                         date_parser = lambda t:pd.to_datetime(t))
        dfs.append(df)
    matches = pd.concat(dfs,ignore_index=True)
    return(matches)

def pull_rankings(start_year, end_year,curr_year):
    dfs = []
    for i in range(int(round(start_year/10)*10), min(int(round(end_year/10)*10),curr_year-1),10):
        print(f'Pulling {i}s historical rankings...')
        url = f'https://github.com/JeffSackmann/tennis_atp/blob/master/atp_rankings_{str(i)[-2:]}s.csv?raw=true'
        df = pd.read_csv(url, index_col=None,
                         header=0,
                         parse_dates=[0],
                         encoding="ISO-8859-1",
                         date_parser=lambda t: pd.to_datetime(t))
        dfs.append(df)
    if end_year >= curr_year:
        print(f'Pulling {curr_year} historical rankings...')
        url = f'https://github.com/JeffSackmann/tennis_atp/blob/master/atp_rankings_current.csv?raw=true'
        df = pd.read_csv(url, index_col=None,
                         header=0,
                         parse_dates=[0],
                         encoding="ISO-8859-1",
                         date_parser=lambda t: pd.to_datetime(t))
        dfs.append(df)
    rankings = pd.concat(dfs, ignore_index=True)
    return (rankings)

def pull_players():
    url = 'https://github.com/JeffSackmann/tennis_atp/blob/master/atp_players.csv?raw=true'
    df = pd.read_csv(url, index_col=None,
                     header=0,
                     parse_dates=[4],
                     encoding="utf-8",
                     date_parser=lambda t: pd.to_datetime(t, errors='coerce'))
    return df

# # pulling ATP match data
start_year = 1968
end_year = 2023
atp_df = pull_atp(start_year,end_year)
atp_df.to_csv(f'{wd}/Data/atp_matches.csv')
print(atp_df)

# pulling futures match data
start_year = 1991
end_year = 2023
futures_df = pull_futures(start_year,end_year)
futures_df.to_csv(f'{wd}/Data/futures_matches.csv')
print(futures_df)

# pulling futures match data
start_year = 1978
end_year = 2023
qual_df = pull_qual(start_year,end_year)
qual_df.to_csv(f'{wd}/Data/qual_challenger_matches.csv')
print(qual_df)

# pulling rankings
curr_year = 2023
start_year = 1970
end_year = 2023
rankings_df = pull_rankings(start_year,end_year,curr_year)
rankings_df.to_csv(f'{wd}/Data/rankings.csv')

# pull player data
player_df = pull_players()
player_df.to_csv(f'{wd}/Data/player_data.csv')