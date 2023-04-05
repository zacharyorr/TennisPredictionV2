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
import xgboost.sklearn as xgb


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

def import_odds(start_year, end_date, start_date):
    dfs = []
    if (end_date.year - start_date.year) > 0:
        for i in range(start_date.year, end_date.year+1):
            print(i)
            odds = pd.read_excel(f'http://www.tennis-data.co.uk/{i}/{i}.xlsx', parse_dates=['Date'])
            dfs.append(odds)
        df = pd.concat(dfs, ignore_index=True)
        df = df[df.Date > start_date]
        return df
    else:
        print(start_date)
        odds = pd.read_excel(f'http://www.tennis-data.co.uk/{start_date.year}/{start_date.year}.xlsx', parse_dates=['Date'])
        df = pd.DataFrame(odds)
        return df

def clean_odds(df, start_date, end_date):
    start_count = df.shape[0]
    df = df[(df.Comment == 'Completed')]
    end_count = df.shape[0]
    print(f'Removed {start_count - end_count} matches that were not completed')

    df['Date'] = pd.to_datetime(df['Date'])

    df = df[(df['Date'] >= start_date) & (df['Date'] < end_date)]
    end2_count = df.shape[0]
    print(f'Removed {end_count - end2_count} matches that were outside the date range')
    print(f'remaining in odds df: {df.shape[0]}')
    return df

def import_players():
    players = pd.read_csv(f'{wd}/Data/player_data.csv', parse_dates=['dob'])
    return players

def import_matches(start_date, end_date, filter_tour_finals, filter_january):
    match_df = pd.read_parquet(f'{wd}/Data/output_df.parquet.gzip')

    for i in range(1,34):
        match_df = match_df.drop([f'p1_seed_{i}'], axis=1)
        match_df = match_df.drop([f'p2_seed_{i}'], axis=1)

    match_df = match_df.drop(['p1_seed_35', 'p2_seed_34', 'p2_seed_35'], axis=1)

    if filter_january == True:
        match_df = match_df[~(match_df.tourney_name.isin(['Australian Open', 'Adelaide 1', 'Adelaide 2', 'Sydney', 'Melbourne']))]

    if filter_tour_finals == True:
        match_df = match_df[~(match_df.tourney_name == 'Tour Finals')]

    # match_df = match_df[~(match_df.tourney_date ==)]

    # match_df['y_pred_xgboost'] = xgboost_inference(match_df, match_df[match_df.tourney_date <= pd.to_datetime(start_year, format='%Y')].count()/match_df.shape[0], 8)

    # match_df = match_df[match_df.ATP == 1]
    #
    # match_df = match_df[(match_df.tourney_date >= start_date) & (match_df.tourney_date < end_date)]

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
    print(f'Accuracy is {(df.y_pred.round() == df.winner).mean()}')
    # df = df[df['best_of_3'] == 0]
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

class Net4(nn.Module):
  def __init__(self,input_shape, input_p, p):
    super(Net4,self).__init__()
    self.fc1 = nn.Linear(input_shape,128)
    self.fc2 = nn.Linear(128,256)
    self.fc3 = nn.Linear(256,512)
    self.fc4 = nn.Linear(512,256)
    self.fc5 = nn.Linear(256,64)
    self.fc6 = nn.Linear(64,16)
    self.fc7 = nn.Linear(16,4)
    self.fc8 = nn.Linear(4, 1)
    self.dropout_input = nn.Dropout(input_p)
    self.dropout = nn.Dropout(p)

  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = self.dropout(x)

    x = torch.relu(self.fc2(x))
    x = self.dropout(x)

    x = torch.relu(self.fc3(x))
    x = self.dropout(x)

    x = torch.relu(self.fc4(x))
    x = self.dropout(x)

    x = torch.relu(self.fc5(x))
    x = self.dropout(x)

    x = torch.relu(self.fc6(x))
    x = self.dropout(x)

    x = torch.relu(self.fc7(x))
    # x = self.dropout(x)

    x = torch.sigmoid(self.fc8(x))
    return x



def load_model(name, model_type):
    model = model_type
    model.load_state_dict(torch.load(f'{wd}/models/{name}.pth'))
    model.eval()

    # for values in model.state_dict():
        # print(values, "\t", model.state_dict()[values])
    return model

def inference(match_df, model, boost_iterations, model_type, nn_iteration_type, start_date, end_date, ATP_only, learning_rate):
    df = match_df.copy()
    # print(df.head(20))
    df = df.drop(['p1_seed', 'p1_ht', 'p2_seed', 'p2_ht', 'p1_rank',
           'p1_rank_points', 'p2_rank', 'p2_rank_points', 'tourney_level_consolidated', 'tourney_code_no_year','total_games_rounded', 'decade', 'total_games'], axis=1)

    # df = df.drop(['dt_index'], axis=1)

    df = df.drop(['BR', 'ER', 'F', 'Q1', 'Q2', 'Q3', 'Q4', 'QF', 'R128', 'R16', 'R32', 'R64', 'RR', 'SF'], axis=1)


    X = df.drop(['index','level_0','tourney_id','tourney_name','surface',
                              'draw_size',
                              'tourney_level',
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
                              # 'tournament_search',
                              # 'city',
                              'city_name',
                              'country_code',
                              'p1_score',
                              'p2_score',
                              'p1_entry',
                              'p2_entry',
                              'winner'
                              ],axis=1)

    X = X.drop(['dt_index'], axis=1)

    # for i in range(1,34):
    #     match_df = match_df.drop([f'p1_seed_{i}'], axis=1)
    #     match_df = match_df.drop([f'p2_seed_{i}'], axis=1)
    #
    # match_df = match_df.drop(['p1_seed_35', 'p2_seed_34', 'p2_seed_35'], axis=1)

    print(X.shape[0])

    num_months = 12 * (end_date.year - pd.to_datetime(start_date).year) + end_date.month - pd.to_datetime(start_date).month

    cutoff_date = start_date

    print(f'Number of periods: {num_months}')
    combined_preds = []
    for i in range(num_months):

        if nn_iteration_type == 'months':
            next_cutoff_date = (pd.to_datetime(cutoff_date) + np.timedelta64(31, 'D')).replace(day=1)

        print(f'Period {i+1}: ({cutoff_date.year}-{cutoff_date.month} to {next_cutoff_date.year}-{next_cutoff_date.month})')
        # print(f'Cutoff date {cutoff_date}')
        # print(f'Cutoff date {next_cutoff_date}')

        X_train = X[X['tourney_date'] < cutoff_date]
        X_test = X[(X['tourney_date'] >= cutoff_date) & (X['tourney_date'] < next_cutoff_date)]

        if ATP_only == 1:
            X_test = X_test[X_test.ATP == 1]

        print(f'Number of test samples: {X_test.shape[0]}')

        X_test = X_test.drop(['tourney_date'], axis=1)
        X_train = X_train.drop(['tourney_date'], axis=1)


        if len(X_test) > 0:
            sc = StandardScaler()
            # print(X_train.head(10))
            # print(X_train.shape[1])
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            model = load_model(f'model_{learning_rate}_{cutoff_date.year}{cutoff_date.month}_{next_cutoff_date.year}{next_cutoff_date.month}', model_type)
            model.eval()
            with torch.no_grad():
                y_pred = model(torch.tensor(X_test, dtype=torch.float32))
            preds = y_pred.detach().numpy()
            for pred in preds:
                combined_preds.append(float(pred))

        cutoff_date = next_cutoff_date

    df['y_pred'] = 0
    print(len(df[(df['tourney_date'] >= start_date) & (df['tourney_date'] < end_date)]))
    print(len(combined_preds))
    df = df[(df['tourney_date'] >= start_date) & (df['tourney_date'] < end_date) & (df['ATP'] == 1)]
    df['y_pred'] = combined_preds
    # print(type(df.loc[0,'y_pred']))
    # print(df.head(40))
    # print(df.dtypes)
    return df

def xgboost_inference(df, train_percent, iterations):

    print('yo')
    X = df.drop(['p1_seed', 'p1_ht', 'p2_seed', 'p2_ht', 'p1_rank',
       'p1_rank_points', 'p2_rank', 'p2_rank_points', 'tourney_level_consolidated', 'tourney_code_no_year','total_games_rounded', 'decade', 'total_games','index','level_0','tourney_id','tourney_name','surface',
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
                          'p2_entry','winner'],axis=1)

    model = xgb.XGBModel()

    for i in range(iterations):
        X_train, X_test = train_test_split(X, train_size=(train_percent+(1-train_percent)*(i/iterations)),
                                                            shuffle=False, random_state=420)

        if i == 0:
            results = np.zeros_like(X_train.shape[0])


        if i + 1 != iterations:
            X_test, X_discard = train_test_split(X_test, train_size=(1) / (iterations - i),
                                                                    shuffle=False, random_state=420)

        # mask = X_test['ATP'] > 0
        # X_test = X_test[mask]

        model.load_model(f'xgboost/xgboost_iteration_{i+1}of{iterations}.json')
        print(model.best_ntree_limit)

        for pred in model.predict(X_test, ntree_limit=model.best_ntree_limit):
            results.append(pred)

    return results


def fractional_kelly(match_df, bankroll, fraction, underdog_threshold, overdog_threshold, graph):
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
        x = print(f'{df.loc[i, "tourney_name"]}, {df.loc[i, "year"]}, {df.loc[i, "round"]}')
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

            f = fraction * (p - (q/b))

            if ((f > 0) & ((b > underdog_threshold) | (b < overdog_threshold))):
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

    if graph == True:
        graphing(bankrolls, dates)

    return bankroll

    # return bankrolls, dates



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
    start_year = 2019
    start_date = pd.to_datetime('20150101',format='%Y%m%d')
    end_year = pd.to_datetime('20180101',format='%Y%m%d')
    odds_df = import_odds(start_year, end_year, start_date)
    # print(odds_df.head(10))
    odds_df = clean_odds(odds_df, start_date, end_year)

    filter_tour_finals = True
    filter_january = True

    match_df = import_matches(start_date, end_year, filter_tour_finals, filter_january)

    player_df = import_players()
    # print(player_df.head(10))
    player_df = index_players(odds_df, player_df)
    # print(player_df.head(10))


    model_name = 'model_4layers_batch64lr0.005_0.68346'
    model_type = Net4(input_shape=109, input_p=0, p=0)
    # model = load_model(model_name, model_type)
    model = ''

    boost_iterations = 6
    nn_iteration_type = 'months'
    ATP_only = 1
    learning_rate = 0.00025
    match_df = inference(match_df, model, boost_iterations, model_type, nn_iteration_type, start_date, end_year, ATP_only, learning_rate)

    match_df = match_index(match_df, odds_df, player_df)


    match_df = betting_calc(match_df)
    bankroll = 1
    final_bankrolls = []
    fractions = [.08]
    # fraction = .1
    underdog_thresholds = [0]
    # underdog_thresholds = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]

    overdog_thresholds = [100]
    # overdog_thresholds = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]

    max_bankroll = 0
    underdog_thresh = 0
    over_thresh = 0

    graph = True
    for fraction in fractions:
        for underdog_threshold in underdog_thresholds:
            for overdog_threshold in overdog_thresholds:
                final_bankroll = fractional_kelly(match_df, bankroll, fraction, underdog_threshold, overdog_threshold, graph)
                final_bankrolls.append(final_bankroll)
                if final_bankroll > max_bankroll:
                    max_bankroll = final_bankroll
                    underdog_thresh = underdog_threshold
                    over_thresh = overdog_threshold

    print(underdog_thresholds)
    print(final_bankrolls)
    print(max_bankroll)
    print(underdog_thresh)
    print(over_thresh)
    # graphing(bankrolls, dates)

main()