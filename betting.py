import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


wd = 'C:/Users/zacha/PycharmProjects/TennisPredictionV2'
os.chdir(wd)
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_rows = 10000

class Net3(nn.Module):
  def __init__(self,input_shape):
    super(Net3,self).__init__()
    self.fc1 = nn.Linear(input_shape,128)
    self.fc2 = nn.Linear(128,256)
    self.fc3 = nn.Linear(256,512)
    self.fc4 = nn.Linear(512,256)
    self.fc5 = nn.Linear(256,64)
    self.fc6 = nn.Linear(64,16)
    self.fc7 = nn.Linear(16,4)
    self.fc8 = nn.Linear(4, 1)

  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = torch.relu(self.fc4(x))
    x = torch.relu(self.fc5(x))
    x = torch.relu(self.fc6(x))
    x = torch.relu(self.fc7(x))
    x = torch.sigmoid(self.fc8(x))
    return x

def import_odds(start_year, end_year, start_date):
    dfs = []
    if (end_year - start_year) > 0:
        for i in range(start_year, end_year):
            odds = pd.read_excel(f'http://www.tennis-data.co.uk/{i}/{i}.xlsx', parse_dates=['Date'])
            dfs.append(odds)
        df = pd.concat(dfs, ignore_index=True)
        df = df[df.Date > start_date]
        return df
    else:
        print(start_year)
        odds = pd.read_excel(f'http://www.tennis-data.co.uk/{start_year}/{start_year}.xlsx', parse_dates=['Date'])
        df = pd.DataFrame(odds)
        return df

def clean_odds(df):
    start_count = df.shape[0]
    df = df[(df.Comment == 'Completed')]
    end_count = df.shape[0]
    print(df.shape[0])

    print(f'Removed {start_count - end_count} matches that were not completed')
    return df

def import_players():
    players = pd.read_csv(f'{wd}/Data/player_data.csv', parse_dates=['dob'])
    return players

def import_matches(start_year):
    match_df = pd.read_parquet(f'{wd}/Data/output_df.parquet.gzip')
    match_df = match_df[(match_df.p1_id == 104925) | (match_df.p2_id == 104925)]

    # match_df['tourney_date'] = match_df['tourney_date']
    # print(f'Match_df shape')
    # print(match_df.shape[0])
    match_df = match_df[match_df.tourney_date >= pd.to_datetime(start_year, format='%Y')]
    # print(match_df.shape[0])
    match_df = match_df[match_df.ATP == 1]
    # print(match_df.shape[0])

    return match_df


def index_players(odds_df, player_df):
    player_df['abb_name'] = player_df['name_last'] + ' ' + player_df['name_first'].astype(str).str[0] + '.'
    print(f'player_df size {player_df.shape[0]}')
    player_df = player_df[player_df.dob > '1970-01-01']
    print(f'player_df size after filtering by dob {player_df.shape[0]}')

    num_duplicates = len(player_df) - len(player_df.drop_duplicates(subset='abb_name'))
    print(f'Number of duplicates: {num_duplicates}')


    return player_df

def match_index(match_df, odds_df, player_df):
    df = match_df.copy()
    mask = (df.winner == 1)
    # print(df.head(10))
    df['winner_name'] = ''
    df['loser_name'] = ''
    df.loc[mask, 'winner_id'] = df.loc[mask, 'p2_id']
    df.loc[mask, 'loser_id'] = df.loc[mask, 'p1_id']
    # print(df.head(10))
    mask = (df.winner == 0)
    df.loc[mask, 'winner_id'] = df.loc[mask, 'p1_id']
    df.loc[mask, 'loser_id'] = df.loc[mask, 'p2_id']
    # print(df.head(10))

    player_df = player_df[['player_id','abb_name']]
    # print(df.shape[0])
    df = df.merge(player_df, how='left', left_on='winner_id', right_on='player_id').drop('player_id',axis=1).rename(columns={'abb_name':'winner_abb_name'})
    df = df.merge(player_df, how='left', left_on='loser_id', right_on='player_id').drop('player_id',axis=1).rename(columns={'abb_name':'loser_abb_name'})

    df = df.set_index('dt_index',drop=True)
    odds_df = odds_df[['Date','Winner','Loser','PSW','PSL']].set_index('Date',drop=True).sort_index()
    print(f'Number of matches in odds df: {odds_df.shape[0]}')

    # print(df.shape[0])
    df = pd.merge_asof(df, odds_df, left_by=['winner_abb_name','loser_abb_name'], right_by=['Winner','Loser'], left_on='dt_index', right_on='Date', direction='nearest', tolerance=pd.Timedelta('7d'))
    # print(df.shape[0])
    df = df.dropna(subset=['PSW','PSL'])
    print(f'Number of matches in odds df: {df.shape[0]}')
    df['ps_winner_odds'] = 1/df['PSW']
    df['ps_loser_odds'] = 1/df['PSL']
    # print(df.shape[0])

    df = df.drop(['winner_abb_name','loser_abb_name','Winner','Loser'],axis=1)

    # print(df.head(50))
    return df

class Net(nn.Module):
  def __init__(self,input_shape):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_shape,128)
    self.fc2 = nn.Linear(128,256)
    self.fc3 = nn.Linear(256,64)
    self.fc4 = nn.Linear(64,4)
    self.fc5 = nn.Linear(4,1)
  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = torch.relu(self.fc4(x))
    x = torch.sigmoid(self.fc5(x))
    return x

class Net2(nn.Module):
  def __init__(self,input_shape):
    super(Net2,self).__init__()
    self.fc1 = nn.Linear(input_shape,128)
    self.fc2 = nn.Linear(128,256)
    self.fc3 = nn.Linear(256,64)
    self.fc4 = nn.Linear(64,32)
    self.fc5 = nn.Linear(32,8)
    self.fc6 = nn.Linear(8,1)
  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = torch.relu(self.fc4(x))
    x = torch.relu(self.fc5(x))
    x = torch.sigmoid(self.fc6(x))
    return x

def load_model(name, model_type):
    model = model_type
    model.load_state_dict(torch.load(f'{wd}/models/{name}.pth'))
    model.eval()

    # for values in model.state_dict():
        # print(values, "\t", model.state_dict()[values])
    return model

def inference(match_df, model):
    df = match_df.copy()
    print(df.head(20))
    X = df.drop(['p1_seed', 'p1_ht', 'p1_age', 'p2_seed', 'p2_ht', 'p2_age', 'p1_rank',
       'p1_rank_points', 'p2_rank', 'p2_rank_points', 'p1_home',
       'p2_home', 'tourney_level_consolidated', 'tourney_code_no_year','total_games_rounded', 'decade', 'total_games','index','level_0','tourney_id','tourney_name','surface',
                          'draw_size',
                          'tourney_level',
                          'tourney_date',
                          'match_num',
                          'p1_id',
                          'p1_name',
                          'p1_hand',
                          # 'p1_ht',
                          'p1_ioc',
                          # 'p1_age',
                          'p2_id',
                          'p2_name',
                          'p2_hand',
                          # 'p2_ht',
                          'p2_ioc',
                          # 'p2_age',
                          'score',
                          'best_of',
                          'round',
                          'minutes',
                          'tournament_search',
                          'city',
                          'country_code',
                          'p1_score',
                          'p2_score',
                          'p1_entry',
                          'p2_entry','winner','winner_name','loser_name','winner_id','loser_id','PSW','PSL','ps_winner_odds','ps_loser_odds'],axis=1)

    sc = StandardScaler()
    # print(X['p1_time_oncourt_last_match'].head(10))
    # var = 'time_oncourt_last_match'
    # X['p1_time_oncourt_last_match']= shuffle(X['p1_time_oncourt_last_match'], random_state=20)
    # for player in ['p1','p2']:
    #     X.loc[:,[f'{player}_{var}']] = shuffle(X.loc[:,[f'{player}_{var}']],random_state=420)
    # print(X['p1_time_oncourt_last_match'].head(10))
    X = sc.fit_transform(X)
    with torch.no_grad():
        y_pred = model(torch.tensor(X, dtype=torch.float32))
    df['y_pred'] = y_pred.detach().numpy()
    return df

def fractional_kelly(match_df, bankroll, fraction):
    bankrolls = []
    dates = []
    df = match_df.copy().reset_index(drop=True)
    count = 0
    correct = 0
    accuracy_correct = 0
    for i in range(df.shape[0]):
        # if i < 200:
        # if df.loc
        print(f'Bet number {i+1}')
        print(f'{df.loc[i, "tourney_name"]}, {df.loc[i, "year"]}, {df.loc[i, "round"]}')
        print(f'Player 1: {df.loc[i, "p1_name"]}')
        print(f'Player 2: {df.loc[i, "p2_name"]}')
        print(f'Player 1 book odds: {df.loc[i, "p1_book_odds"]:.2f}')
        print(f'Player 2 book odds: {df.loc[i, "p2_book_odds"]:.2f}')
        print(f'Player 1 model odds: {df.loc[i, "y_pred_p1"]:.2f}')
        print(f'Player 2 model odds: {df.loc[i, "y_pred"]:.2f}')
        if df.loc[i, 'winner'] == 0:
            print(f'Player 1 won by a score of: {df.loc[i, "score"]}')
        if df.loc[i, 'winner'] == 1:
            print(f'Player 2 won by a score of: {df.loc[i, "score"]}')

        print(f'Starting bankroll: {bankroll:.2f}')

        for player in ['p1','p2']:
            p = df.loc[i, f'y_pred_{player}']
            # print(p)
            q = 1-p
            # print(q)
            b = (df.loc[i, f'{player}_decimal_odds'] - 1)
            # print(f'{player} decimal odds: {df.loc[i, f"{player}_decimal_odds"]}')
            # print(b)
            f = fraction * (p - (q/b))
            # print(f'{player} fractional Kelly ratio: {f:.2f}')

            if f > 0:
                count += 1
                bet_size = f * bankroll
                print(f'Betting {bet_size:.2f} on {player}')
                start_br = bankroll
                bankroll = bankroll - bet_size + bet_size * (df.loc[i, f'{player}_winner'] * (df.loc[i, f'{player}_decimal_odds']))
                if (bankroll - start_br) > 0:
                    correct += 1
            else:
                print('No bet placed')
        print(f'Ending bankroll: {bankroll:.2f}\n')

        dates.append(df.loc[i,'tourney_date'])
        bankrolls.append(bankroll)
    print(f'% winning bets: {correct*100/count:.2f}%\n')
    return bankrolls, dates




def betting_calc(match_df):
    df = match_df.copy()

    mask = (df.winner == 1)
    # print(df.head(10))
    df['p1_book_odds'] = 0
    df['p2_book_odds'] = 0
    df['p1_winner'] = 0
    df['p2_winner'] = 0

    df.loc[mask, 'p1_book_odds'] = df.loc[mask, 'ps_loser_odds']
    df.loc[mask, 'p2_book_odds'] = df.loc[mask, 'ps_winner_odds']
    df.loc[mask, 'p1_decimal_odds'] = df.loc[mask, 'PSL']
    df.loc[mask, 'p2_decimal_odds'] = df.loc[mask, 'PSW']
    df.loc[mask, 'p1_winner'] = 0
    df.loc[mask, 'p2_winner'] = 1

    # print(df.head(10))
    mask = (df.winner == 0)
    df.loc[mask, 'p1_book_odds'] = df.loc[mask, 'ps_winner_odds']
    df.loc[mask, 'p2_book_odds'] = df.loc[mask, 'ps_loser_odds']
    df.loc[mask, 'p1_decimal_odds'] = df.loc[mask, 'PSW']
    df.loc[mask, 'p2_decimal_odds'] = df.loc[mask, 'PSL']
    df.loc[mask, 'p1_winner'] = 1
    df.loc[mask, 'p2_winner'] = 0
    # print(df.head(10))

    df['y_pred_p1'] = 1 - df['y_pred']
    df['y_pred_p2'] = df['y_pred']

    return df

def graphing(bankrolls, dates):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Bet Number')
    ax1.set_ylabel('Bankroll', color=color)
    ax1.plot(bankrolls, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    fig.savefig(f'{wd}/bankrolls.png')


def main():
    start_year = 2022
    start_date = pd.to_datetime('20220601',format='%Y%m%d')
    end_year = 2022
    odds_df = import_odds(start_year, end_year, start_date)
    # print(odds_df.head(10))
    odds_df = clean_odds(odds_df)
    match_df = import_matches(start_year)
    # print(match_df.head(50))
    # match_df = match_index(match_df, odds_df)

    player_df = import_players()
    # print(player_df.head(10))
    player_df = index_players(odds_df, player_df)
    # print(player_df.head(10))

    match_df = match_index(match_df, odds_df, player_df)

    model_name = 'model_2_26_0.09'
    model_type = Net3(input_shape=105)
    model = load_model(model_name, model_type)

    match_df = inference(match_df, model)
    match_df = betting_calc(match_df)
    bankroll = 1
    fraction = .033
    bankrolls, dates = fractional_kelly(match_df, bankroll, fraction)
    graphing(bankrolls, dates)

main()