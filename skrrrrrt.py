import os
import random
import numpy as np
import pycountry_convert
import requests
import urllib
import pandas as pd
import geonamescache
from unidecode import unidecode
from glicko import *
import unittest
import datetime
import math
import time
import winsound

wd = 'C:/Users/zacha/PycharmProjects/TennisPrediction'
os.chdir(wd)
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_rows = 10000


def read_files():
    # imports and returns the five raw data files pulled from github.
    # full credit to Jeff Sackmann
    atp_df = pd.read_csv(f'{wd}/Data/atp_matches.csv',
                         index_col=0,
                         header=0,
                         )
    futures_df = pd.read_csv(f'{wd}/Data/futures_matches.csv',
                             index_col=0,
                             header=0,
                             dtype={5: str}
                             )
    qual_challenger_df = pd.read_csv(f'{wd}/Data/qual_challenger_matches.csv',
                                     index_col=0,
                                     header=0,
                                     )
    rankings_df = pd.read_csv(f'{wd}/Data/rankings.csv',
                              index_col=0,
                              header=0,
                              )
    player_df = pd.read_csv(f'{wd}/Data/player_data.csv',
                            index_col=0,
                            header=0,
                            )
    return atp_df, futures_df, qual_challenger_df, rankings_df, player_df


def combine_dfs(atp_df, futures_df, qual_challenger_df):
    # creates an empty list to store the 3 dataframes in, to later concat
    dfs = []
    # appends the 3 existing dataframes to the dfs list
    for arg in (atp_df, futures_df, qual_challenger_df):
        dfs.append(arg)
    # first, confirms that all 3 dataframes are of the same shape, then appends them
    if atp_df.shape[1] == futures_df.shape[1] and atp_df.shape[1] == qual_challenger_df.shape[1]:
        match_df = pd.concat(dfs, ignore_index=True)
        match_df = match_df.sort_values(by=['tourney_date'], ascending=True).reset_index()

        return match_df
    else:
        print('Error. The 3 dataframes do not have the same shape')
        return

def remove_walkovers(match_df):
    # removing walkovers is important as they would heavily skew the data
    # count is later used to determine the number of walkovers/retirements that were removed from the dataset
    count = match_df.shape[0]
    match_df = match_df.copy()

    # uses vectorized .str function to eliminate all rows whose score contains "RET" or "W/O"
    match_df = match_df[~match_df['score'].str.contains('[A-Za-z]|\\?|0-0|>', na=True, case=False, regex=True)]
    match_df.loc[:,'score'] = match_df.loc[:,'score'].str.replace('  ',' ')

    # returns the number of walkovers removed from the dataset
    count = count - match_df.shape[0]
    print(
        f'\nRemoved {count} walkovers and retirements from the dataset.\nThis represents {100 * count / (match_df.shape[0] + count):.2f}% of the original dataset.')
    return match_df

def split_df(match_df):
    print('\nSplitting dataframe...')
    # currently, the winning player is always player 1. In this condition, the dataframe would be found to be biased
    # by the neural network. Thus, we have to split the dataframe, allowing player 1 to be the winner half the time,
    # and player 2 to be the winner half of the time.

    # Creates a 'split' key column, defining whether player 0 or 1 will be the winner
    match_df['split_var'] = np.random.randint(2, size=match_df.shape[0])

    # The next 4 lines are simply for user clarity. They display how many rows were assigned P1 and P2 as winners
    count_zero = match_df['split_var'][match_df['split_var'] == 0].count()
    count_one = match_df['split_var'][match_df['split_var'] == 1].count()
    print(f'setting {count_zero} matches to P1 winner.')
    print(f'setting {count_one} matches to P2 winner.')

    # Creates 2 dataframes to do modification on; one is all the rows that player 1 is the winner, and the other is
    # the dataframe where player 2 is the winner
    p1_winner = match_df[match_df['split_var'] == 0]
    p2_winner = match_df[match_df['split_var'] == 1]

    # Making a copy to prevent the Copy error
    p2_winner_copy = p2_winner.copy(deep=True)

    # Moves the columns around for P2 winner dataframe. Moves all columns that start with P1 to the columns that start
    # with P2
    for arg in [0, 1]:
        list = ['winner_', 'loser_']
        list1 = ['loser_', 'winner_']
        for i in ['id', 'seed', 'entry', 'name', 'hand', 'ht', 'ioc', 'age']:
            p2_winner_copy.loc[:, f'{list[arg]}{i}'] = p2_winner.loc[:, f'{list1[arg]}{i}']

    # Creates an empty list, appends both of the split dataframes, then concatenates them
    combined_df = []
    combined_df.append(p1_winner)
    combined_df.append(p2_winner_copy)
    match_df = pd.concat(combined_df, axis=0).sort_index()

    return match_df

def rename_columns(match_df):
    # Just some housekeeping, changing the syntax of a bunch of columns now that P1 is not always the winner.
    # Wasn't done manually, did most with ChatGPT
    match_df = match_df.rename(columns={
        'winner_id': 'p1_id',
        'winner_seed': 'p1_seed',
        'winner_entry': 'p1_entry',
        'winner_name': 'p1_name',
        'winner_hand': 'p1_hand',
        'winner_ht': 'p1_ht',
        'winner_ioc': 'p1_ioc',
        'winner_age': 'p1_age',

        'loser_id': 'p2_id',
        'loser_seed': 'p2_seed',
        'loser_entry': 'p2_entry',
        'loser_name': 'p2_name',
        'loser_hand': 'p2_hand',
        'loser_ht': 'p2_ht',
        'loser_ioc': 'p2_ioc',
        'loser_age': 'p2_age',
        # 'split_var': 'winner',

        'w_ace': 'p1_ace',
        'w_df': 'p1_df',
        'w_svpt': 'p1_svpt',
        'w_1stIn': 'p1_1stIn',
        'w_1stWon': 'p1_1stWon',
        'w_2ndWon': 'p1_2ndWon',
        'w_SvGms': 'p1_SvGms',
        'w_bpSaved': 'p1_bpSaved',
        'w_bpFaced': 'p1_bpFaced',

        'l_ace': 'p2_ace',
        'l_df': 'p2_df',
        'l_svpt': 'p2_svpt',
        'l_1stIn': 'p2_1stIn',
        'l_1stWon': 'p2_1stWon',
        'l_2ndWon': 'p2_2ndWon',
        'l_SvGms': 'p2_SvGms',
        'l_bpSaved': 'p2_bpSaved',
        'l_bpFaced': 'p2_bpFaced',

        'winner_rank': 'p1_rank',
        'winner_rank_points': 'p1_rank_points',
        'loser_rank': 'p2_rank',
        'loser_rank_points': 'p2_rank_points'

    })
    return match_df

def remove_features(match_df):
    print('\nRemoving unneeded features...')
    df = match_df.copy()
    df = df.drop(['p1_ace',
                  'p1_df',
                  'p1_svpt',
                  'p1_1stIn',
                  'p1_1stWon',
                  'p1_2ndWon',
                  'p1_SvGms',
                  'p1_bpSaved',
                  'p1_bpFaced',
                  'p2_ace',
                  'p2_df',
                  'p2_svpt',
                  'p2_1stIn',
                  'p2_1stWon',
                  'p2_2ndWon',
                  'p2_SvGms',
                  'p2_bpSaved',
                  'p2_bpFaced'
                  ],axis=1)
    return df

def pull_cities(tourney_list):
    print('Parsing tournament names from dataset...')
    # Creates two empty lists that will later be filled with indexed cities and countries
    country_list = np.empty_like(tourney_list)
    city_list = np.empty_like(tourney_list)

    # Initiates the geonamescache module; used to parse location data
    gc = geonamescache.GeonamesCache()
    cities = gc.get_cities()

    # Creates a dictionary of the cities in the database. The cities must have a length greater than 3 (to prevent false
    # positives) and are then sorted by length, population
    sorted_cities = dict(
        filter(lambda x: len(x[1]["name"]) > 3,
               sorted(cities.items(), key=lambda x: (len(x[1]["name"]), x[1]["population"]), reverse=True)))

    # Moves from a dictionary to a DataFrame. Slower, but easier to modify
    country_df = pd.DataFrame.from_dict({(i): sorted_cities[i]
                                         for i in sorted_cities.keys()},
                                        orient='index')

    # Changes all cities to ASCII and makes them lowercase
    for i in range(len(country_df["name"])):
        country_df["name"][i] = unidecode(country_df["name"][i]).lower()

    # Adds 5 columns to the dataframe, one for each of the first 5 alternate names of the corresponding city
    for i in range(5):
        country_df[f"altname{i + 1}"] = country_df["alternatenames"].str[i]
        country_df[f"altname{i + 1}"] = country_df[f"altname{i + 1}"].str.lower().astype(str)
        country_df[f'altname{i + 1}_len'] = country_df[f"altname{i + 1}"].str.len()

    # Drops unnecessary columns, and filters for a population greater than 15,000
    country_df = country_df.reset_index().drop(
        ['index', 'geonameid', 'latitude', 'longitude', 'timezone', 'admin1code', 'alternatenames'], axis=1)
    country_df.population = country_df.population.astype(int)
    country_df = country_df[country_df.population > 15000].reset_index()

    tourney_count = 0
    # to fix, fix the ch, now it gets rid of ho chi minh city
    # also, make second dataframe and check first in the for loop for exact matches
    tourney_list = [x.lower().replace(' ch', '')
                    .replace(' 1', '')
                    .replace(' 2', '')
                    .replace(' 3', '')
                    .replace(' 4', '')
                    .replace(' 5', '')
                    .replace(' 6', '')
                    .replace(' 7', '')
                    .replace(' 8', '')
                    .replace(' 9', '')
                    .replace('1', '')
                    .replace('2', '')
                    .replace('3', '')
                    .replace('4', '')
                    .replace('5', '')
                    .replace('6', '')
                    .replace('7', '')
                    .replace('8', '')
                    .replace('9', '')
                    for x in tourney_list]

    # Loops through all tournaments in the tournament list
    for tourney in tourney_list:
        # Loops through all rows in country_df
        for i in range(len(country_df['name'])):
            # Edge case handling, Davis Cup and Tour Finals produce errors (country Davis), so they are removed
            if (('davis cup' in tourney) or
                    ('tour finals' in tourney)):
                # Break inner for loop if condition is met
                break

            # If the tournament is not one of the exceptions, checks if any cities or their alternate names match
            elif (
                    (country_df.name[i] in tourney) or
                    (len((country_df.altname1[i])) > 4 and (country_df.altname1[i]) in tourney) or
                    (len((country_df.altname2[i])) > 4 and (country_df.altname2[i]) in tourney) or
                    (len((country_df.altname3[i])) > 4 and (country_df.altname3[i]) in tourney) or
                    (len((country_df.altname4[i])) > 4 and (country_df.altname4[i]) in tourney)
            ):
                # Adds that city/country pair into the appropriate lists
                city_list[tourney_count] = (country_df.name[i])
                country_list[tourney_count] = (country_df.countrycode[i])
                break

        if tourney_count % 100 == 0:
            print(tourney_count)
        tourney_count += 1

    # Creates a DataFrame from the parsed countries
    data = {'tourney': tourney_list, 'city': city_list, 'country': country_list}
    df = pd.DataFrame(data)

    # Converts the existing Alpha-2 country code to Alpha-3
    df['country_code'] = ""
    for i in range(len(df.country)):
        if df.country[i] != 'None':
            try:
                # Uses pycountry_convert. If the country is "None", then the exception is passed
                country = pycountry_convert.country_alpha2_to_country_name(df.country[i])
                df['country_code'][i] = pycountry_convert.country_name_to_country_alpha3(country)
            except KeyError:
                print(f"Invalid Country Alpha-2 code: {df.country[i]}")
        else:
            print(f"Country code is None for index {i}")

    df.to_csv(f'{wd}/Data/country_mapping.csv')

def get_tournaments(match_df):
    # Precursor to pulling cities method, molds the data according to the needs of the city parser and wikipedia parser
    match_df['tournament_search'] = match_df['tourney_name'] + ' tennis tournament ' + match_df['tourney_date'].astype(
        str).str[:4]
    tourney_list = match_df['tournament_search'].unique()
    no_year_tourney_list = match_df['tourney_name'].unique()
    return tourney_list, no_year_tourney_list

def attach_tournaments(match_df, tourney_list):
    print('\nAttaching tournament names...')
    # Puts the calculated countries into the match dataframe. Starts by reading the exported CSV from pull_cities
    country_df = pd.read_csv(f'{wd}/Data/country_mapping.csv',
                             index_col=0,
                             header=0,
                             )
    # Undoes the string manipulation we performed earlier, for indexing
    country_df['original_tourney_name'] = tourney_list.astype(str)
    # Drops unnecessary columns
    country_df = country_df.drop(['tourney', 'country'], axis=1)
    # Left outer join to match the appropriate country codes
    result_df = match_df.join(country_df.set_index('original_tourney_name'), on='tourney_name', validate='m:1')
    return result_df

def one_hot_bof5(df):
    # Converts the Best of 5, Country Codes, and Hand features to one_hot features for the NN

    # Eliminates exhibition matches
    df = df[df['best_of'] != 1]

    # Creates dummies for the best of 3 column, and adds those columns to the Dataframe
    one_hot = pd.get_dummies(df['best_of'])
    df = df.join(one_hot[3])
    df = df.rename({3: 'best_of_3'}, axis=1)

    # Calculates whether P1/P2 have home court advantage, puts that into a one_hot variable
    df['p1_home'] = (df['p1_ioc'] == df['country_code']).astype(int)
    df['p2_home'] = (df['p2_ioc'] == df['country_code']).astype(int)

    # The following code converts handedness into one_hot variables, L, R, or U (unknown)
    df = df[df['p1_hand'] != 'A']
    one_hot = pd.get_dummies(df['p1_hand'])
    df = df.join(one_hot[['L', 'R']])
    df = df.rename({'L': 'p1_left_handed', 'R': 'p1_right_handed'}, axis=1)

    # Repeats for P2
    df = df[df['p2_hand'] != 'A']
    one_hot = pd.get_dummies(df['p2_hand'])
    df = df.join(one_hot[['L', 'R']])
    df = df.rename({'L': 'p2_left_handed', 'R': 'p2_right_handed'}, axis=1)

    return df

def set_h2h(match_df, rankings_df):
    # Sorts the match DataFrame by tournament date. Head to head is date-sensitive so this is needed
    temp_df = match_df.sort_values(by=['tourney_date'], ascending=True).reset_index()

    # Creates columns for the H2H
    temp_df['p1_h2h_wins'] = 0
    temp_df['p2_h2h_wins'] = 0

    # Creates new dataframe to store and update H2H values for efficiency
    df = temp_df[['p1_id', 'p2_id']]
    # Sorts the ID's, so that players appearing in opposite order aren't counted twice
    df = df.copy()
    df.loc[:, 'key1'] = df.loc[:, ['p1_id', 'p2_id']].min(axis=1)
    df.loc[:, 'key2'] = df.loc[:, ['p1_id', 'p2_id']].max(axis=1)
    df['composite_key'] = df['key1'].astype(str) + '-' + df['key2'].astype(str)
    # Sets the composite key in the main DataFrame
    temp_df['h2h_key'] = df['composite_key']

    # Keeps only unique H2H matchup keys, and sets that as the index
    df = df.drop_duplicates(subset=['composite_key'], keep='first')
    df = df.set_index('composite_key')
    df = df.sort_index()

    # Drops unnecessary columns and sets starting values to 0
    df = df.drop(['p1_id', 'p2_id'], axis=1)
    df['p1_wins'] = 0
    df['p2_wins'] = 0
    numrows = temp_df.shape[0]
    match_df = temp_df.copy(deep=True)

    # Convert to Dict
    h2h_dict = df.to_dict()
    match_dict = temp_df.to_dict()

    print('\nCreating historical head-to-head records...')
    for i in range(numrows):
        index = match_dict['h2h_key'][i]
        winner = match_dict['p1_id'][i]
        if h2h_dict['key1'][index] == winner:
            match_dict['p1_h2h_wins'][i] = h2h_dict['p1_wins'][index]
            match_dict['p2_h2h_wins'][i] = h2h_dict['p2_wins'][index]
            h2h_dict['p1_wins'][index] += 1
        else:
            match_dict['p1_h2h_wins'][i] = h2h_dict['p2_wins'][index]
            match_dict['p2_h2h_wins'][i] = h2h_dict['p1_wins'][index]
            h2h_dict['p2_wins'][index] += 1


        if (i + 1) % 50000 == 0:
            print(f'{i + 1} of {numrows} head-to-head records created')

    match_df['p1_h2h_wins'] = match_dict['p1_h2h_wins']
    match_df['p2_h2h_wins'] = match_dict['p2_h2h_wins']
    match_df = match_df.drop(['index', 'h2h_key'], axis=1)
    return match_df


def total_games(row):
    """this function takes the input row and strips the score string, parsing the number of games in each set,
    and adding these together, returning the sum"""
    total_games = 0
    if row is not None:
        try:
            p1_games = int(row.split('-')[0])
            if len(row.split('-')[1]) >=2:
                if row.split('-')[1][1] == '(':
                    p2_games = int(row.split('-')[1][0])
                else:
                    p2_games = int(row.split('-')[1][0:2])
            else:
                p2_games = int(row.split('-')[1][0])
            total_games = p1_games + p2_games
        except:
            print(row)
    return total_games


def p2_games(row):
    """this function takes the input row and strips the score string, parsing the number of games in each set,
    and adding these together, returning the sum. a similar calculation is done in total_games"""
    p2_games = 0
    if row is not None:
        try:
            if len(row.split('-')[1]) >=2:
                if row.split('-')[1][1] == '(':
                    p2_games = int(row.split('-')[1][0])
                else:
                    p2_games = int(row.split('-')[1][0:2])
            else:
                p2_games = int(row.split('-')[1][0])
        except:
            print(row)
    return p2_games


def p1_games(row):
    """this function takes the input row and strips the score string, parsing the number of games in each set,
    and adding these together, returning the sum. a similar calculation is done in total_games"""
    p1_games = 0
    if row is not None:
        try:
            p1_games = int(row.split('-')[0])
        except:
            print(row)
    return p1_games

def p1_score_calc(row, sets_weight, games_weight):
    """this function uses the 3 components of the score (outcome, sets won, and games won) to calculate the score
    for player 1 of that match"""
    score = row.p1_score
    sets_won = 0
    for i in range(row.num_sets):
        score += (row[f'set_{i+1}_games_p1'] / row[f'set_{i+1}_games']) * (games_weight/row.num_sets)
        if row[f'set_{i+1}_games_p1'] > row[f'set_{i+1}_games_p2']:
            sets_won+= 1
    score += (sets_weight * sets_won) / row.num_sets
    return score

def p2_score_calc(row, sets_weight, games_weight):
    score = row.p2_score
    sets_won = 0
    for i in range(row.num_sets):
        score += (row[f'set_{i+1}_games_p2'] / row[f'set_{i+1}_games']) * (games_weight/row.num_sets)
        if row[f'set_{i+1}_games_p2'] > row[f'set_{i+1}_games_p1']:
            sets_won += 1
    score += (sets_weight * sets_won) / row.num_sets
    return score


def winner_points(match_df,winner_score, sets_weight, games_weight):
    """this function calculates the score of the winner and loser of each match. I use custom weights to calculate
    the true score of the match. currently, out of 1 possible point, .6 is allocated to the winner, .2 is distributed
    based on number of sets won, and .2 is distributed based on number of games won

    these scores feed directly into the glicko ratings but are not fed into the NN"""
    temp_df = match_df.copy()
    count = temp_df.shape[0]
    temp_df = temp_df.dropna(subset=['score']).reset_index()
    rows_removed = count - temp_df.shape[0]
    print(f'Rows removed due to invalid scores: {rows_removed}')
    strip = temp_df['score'].str.strip()
    df = strip.str.split(pat=' ',expand=True)

    df['p1_score'] = winner_score
    df['p2_score'] = 0
    df['num_sets'] = df[[0,1,2,3,4]].count(axis=1)
    df['total_games'] = 0

    # this for loop strips the score string and parses it to calculate the total number of games in each set/match
    for i in range(5):
        df[i] = df[i].str.replace('[','',regex=False)
        df[i] = df[i].str.replace(']','',regex=False)
        df[f'set_{i+1}_games'] = df[i].apply(total_games)
        df[f'set_{i+1}_games_p1'] = df[i].apply(p1_games)
        df[f'set_{i+1}_games_p2'] = df[i].apply(p2_games)

    # adding up calculated games to input total games into the dataframe. this is to calculate average time played in
    # the time_on_court section
    df['total_games'] = df['set_1_games'] + df['set_2_games'] + df['set_3_games'] + df['set_4_games'] + df['set_5_games']

    # uses score_calc functions to calculate each player's respective match score
    df['p2_score'] = df.apply(p2_score_calc, args=(sets_weight, games_weight), axis=1)
    df['p1_score'] = df.apply(p1_score_calc, args=(sets_weight, games_weight), axis=1)

    # adds these results back to the original dataframe and returns the updated dataframe
    temp_df['p1_score'] = df['p1_score']
    temp_df['p2_score'] = df['p2_score']
    temp_df['total_games'] = df['total_games']

    return temp_df


def gluck_gluck(match_df, initial_rating, initial_deviation, initial_volatility, rating_period):
    """this function is no longer in use. it calculates the glicko rankings for every match played and adds these
    live ratings to the dataframe. I've switched to Glicko-2 to implement a penalty for long breaks of inactivity"""
    df = match_df.copy()
    s1 = pd.Series(df.loc[:, 'p1_id'])
    s2 = pd.Series(df.loc[:, 'p2_id'])
    df = pd.concat([s1, s2], ignore_index=True).drop_duplicates(keep='first').reset_index()
    # print(df.head(50))
    df.loc[:, 'rating'] = initial_rating
    df.loc[:, 'deviation'] = initial_deviation
    df.loc[:, 'volatility'] = initial_volatility
    df = df.set_index(0, drop=False)
    df = df.drop(['index'], axis=1)
    # print(df.head(50))

    df['opponent_elo'] = np.empty((len(df), 0)).tolist()
    df['opponent_deviation'] = np.empty((len(df), 0)).tolist()
    df['opponent_volatility'] = np.empty((len(df), 0)).tolist()
    df['outcome'] = np.empty((len(df), 0)).tolist()
    df['active'] = 0
    # print(df.head(50))
    ranking_df = df


    start_date = match_df.copy()['tourney_date'].min()
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    # print(start_date)
    rp = int(rating_period)

    glicko_dict = df.copy().to_dict(orient='index')
    ratings_dict = df.copy().to_dict(orient='index')
    match_dict = match_df.copy().to_dict(orient='index')

    # set all key:value pairs to Player instances
    for key in enumerate(df.copy().index):
        glicko_dict[key[1]] = Player()


    rate_period_date = start_date

    while rate_period_date < datetime.datetime.strptime(match_df.copy()['tourney_date'].max(), '%Y-%m-%d'):
        future_period_date = rate_period_date + datetime.timedelta(days=rp)
        filtered_dict = {k: v for k, v in match_dict.items() if (
                    rate_period_date <= datetime.datetime.strptime(v['tourney_date'], '%Y-%m-%d') < future_period_date)}

        for k, v in filtered_dict.items():
            p1_id = v['p1_id']
            p1_elo = glicko_dict[p1_id].rating
            p1_deviation = glicko_dict[p1_id].rd
            p1_volatility = glicko_dict[p1_id].vol

            p2_id = v['p2_id']
            p2_elo = glicko_dict[p2_id].rating
            p2_deviation = glicko_dict[p2_id].rd
            p2_volatility = glicko_dict[p2_id].vol

            winner = v['winner']

            # ratings_dict[p1_id]['opponent_elo'].append(p2_elo)
            # ratings_dict[p1_id]['opponent_deviation'].append(p2_deviation)
            # ratings_dict[p1_id]['opponent_volatility'].append(p2_volatility)
            # ratings_dict[p2_id]['opponent_elo'].append(p1_elo)
            # ratings_dict[p2_id]['opponent_deviation'].append(p1_deviation)
            # ratings_dict[p2_id]['opponent_volatility'].append(p1_volatility)
            if int(winner) == 0:
                # ratings_dict[p1_id]['outcome'].append(v['winner_score'])
                # ratings_dict[p2_id]['outcome'].append(v['loser_score'])
                ratings_dict[p1_id]['outcome'] = (v['winner_score'])
                ratings_dict[p2_id]['outcome'] = (v['loser_score'])
            else:
                # ratings_dict[p1_id]['outcome'].append(v['loser_score'])
                # ratings_dict[p2_id]['outcome'].append(v['winner_score'])
                ratings_dict[p1_id]['outcome'] = (v['loser_score'])
                ratings_dict[p2_id]['outcome'] = (v['winner_score'])
            # ratings_dict[p1_id]['active'] = 1
            # ratings_dict[p2_id]['active'] = 1

            if p1_id == 104745:
                print(
                    f'{p1_id}, rating {glicko_dict[p1_id].rating}, RD {glicko_dict[p1_id].rd}, vol {glicko_dict[p1_id].vol}')
                print(f'Update player count: {glicko_dict[p1_id].counta}')
                print(f'rating list: {p2_elo}')
                print(f'rd list: {p2_deviation}')
                print(f'vol list: {p2_volatility}')
                print(f'outcome list: {ratings_dict[p1_id]["outcome"]}')
                print(f'--------------------')
            if p2_id == 104745:
                print(f'Update player count: {glicko_dict[p2_id].counta}')
                print(
                    f'{p2_id}, rating {glicko_dict[p2_id].rating}, RD {glicko_dict[p2_id].rd}, vol {glicko_dict[p2_id].vol}')
                print(f'rating list: {p1_elo}')
                print(f'rd list: {p1_deviation}')
                print(f'vol list: {p1_volatility}')
                print(f'outcome list: {ratings_dict[p2_id]["outcome"]}')
                print(f'--------------------')
            glicko_dict[p1_id].update_player([p2_elo], [p2_deviation], [ratings_dict[p1_id]['outcome']])
            glicko_dict[p2_id].update_player([p1_elo], [p1_deviation], [ratings_dict[p2_id]['outcome']])
            if p1_id == 104745:
                print(
                    f'{p1_id}, rating {glicko_dict[p1_id].rating}, RD {glicko_dict[p1_id].rd}, vol {glicko_dict[p1_id].vol}')
            if p2_id == 104745:
                print(
                    f'{p2_id}, rating {glicko_dict[p2_id].rating}, RD {glicko_dict[p2_id].rd}, vol {glicko_dict[p2_id].vol}')
                print(f'rating list: {p1_elo}')



        print(f'Period start date: {rate_period_date}')
        print('------------------------')
        rate_period_date = future_period_date


    df['glicko_1'] = 0
    counter = 0
    for i in enumerate(df.copy().index):
        df.loc[counter,'glicko_1'] = glicko_dict[i[1]].rating
        df.loc[counter,'id'] = i[1]
        counter += 1
    df = df.drop(columns=[0])
    df = df.join(match_df.copy()[['p1_id','p1_name']].set_index('p1_id',drop=True), on='id', how='left').drop_duplicates(subset='id',keep='first').reset_index()
    df = df.sort_values(by='glicko_1',ascending=False)
    df = df.set_index(0,drop=True)
    df['index'] = np.arange(len(df))
    df = df.set_index('index',drop=True)
    df = df.drop(['rating','deviation','volatility','opponent_elo','opponent_deviation','opponent_volatility','outcome','active'],axis=1)


def gluck_gluck2(match_df, initial_rating, initial_deviation, initial_volatility, rating_period, show_rankings, rd_cutoff):
    print(f'\nCalculating Glicko-2 ratings...')
    # This code section sets up the output dataframe to be populated later.
    output_df = match_df.copy(deep=True)
    output_df['p1_glicko_rating'] = 0
    output_df['p2_glicko_rating'] = 0
    output_df['p1_glicko_deviation'] = 0
    output_df['p2_glicko_deviation'] = 0
    output_df['p1_glicko_volatility'] = 0
    output_df['p2_glicko_volatility'] = 0

    # This section creates a list of unique player ID's from all matches played in match_df
    # Time-dependent Glicko ratings will be stored in this dataframe (later dictionary)
    df = match_df.copy()
    s1 = pd.Series(df.loc[:, 'p1_id'])
    s2 = pd.Series(df.loc[:, 'p2_id'])
    df = pd.concat([s1, s2], ignore_index=True).drop_duplicates(keep='first').reset_index()
    df = df.set_index(0, drop=False)
    df = df.drop(['index'], axis=1)
    print(f'Glicko ranking database created...')

    # Setting each row value to an empty list. These lists will be populated with each player's opponents' Glicko stats
    # They will then be wiped and re-populated in each defined rating period
    df['opponent_elo'] = np.empty((len(df), 0)).tolist()
    df['opponent_deviation'] = np.empty((len(df), 0)).tolist()
    df['outcome'] = np.empty((len(df), 0)).tolist()
    df['active'] = 0

    # Pulls the start date for the data from the match dataframe
    start_date = match_df.copy()['tourney_date'].min()
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    # In Glicko-2, the rating period is the time period in which all games that occur in that period are considered
    # to have occurred simultaneously (to determine rating standard deviation/volatility)
    rp = int(rating_period)

    # Converting all dataframes to dictionaries (for computational speed)
    glicko_dict = df.copy().to_dict(orient='index')
    ratings_dict = df.copy().to_dict(orient='index')
    match_dict = output_df.to_dict(orient='index')
    print(f'Dictionaries created...\n')

    # Creates a unique Glicko player for each unique player ID in the DataFrame
    print(f'Creating Glicko players...\nInitial Rating: {initial_rating}\nInitial Deviation: {initial_deviation}\nVolatility: {initial_volatility}')
    for key in enumerate(df.copy().index):
        glicko_dict[key[1]] = Player(initial_rating, initial_deviation, initial_volatility)
    print(f'\nGlicko players created...')
    rate_period_date = start_date
    print(f'\nSolving for Glicko ratings, period = {rating_period} days')
    print(f'Start date: {rate_period_date}\n')

    # To compute expected runtime
    start_time = time.time()
    num_loops = ((datetime.datetime.strptime(match_df.copy()['tourney_date'].max(), '%Y-%m-%d')-start_date)/rp).days + 1
    loops_complete = 0


    """ This while loop loops through the match DataFrame multiple times (depending on the length of the rating period
    and the time period covered in the DataFrame. For example, if the DataFrame covers 360 days of matches and the 
    rating period is set to 90 days, then this while loop will occur 4 times. While this procedure is not necessary
    to compute Glicko-1 scores or ELO, it is necessary to calculate Glicko-2"""
    while rate_period_date < datetime.datetime.strptime(match_df.copy()['tourney_date'].max(), '%Y-%m-%d'):
        # Sets an end date for this period (current day + rating period). Ex: (Jan 1 + 40 days) = ~Feb 10
        future_period_date = rate_period_date + datetime.timedelta(days=rp)
        # Filters the match DataFrame for all matches that occur in this time period
        filtered_dict = {k: v for k, v in match_dict.items() if (
                rate_period_date <= datetime.datetime.strptime(v['tourney_date'], '%Y-%m-%d') < future_period_date)}

        # Loops through this newly created filtered match dictionary
        for k, v in filtered_dict.items():
            # Setting p1/p2 values for later manipulation
            p1_id = v['p1_id']
            p1_elo = glicko_dict[p1_id].rating
            p1_deviation = glicko_dict[p1_id].rd

            p2_id = v['p2_id']
            p2_elo = glicko_dict[p2_id].rating
            p2_deviation = glicko_dict[p2_id].rd

            """Appends each match opponent's statistics onto their opponent's lists. For example, if P1 plays P2,
            (rating 1400, RD 200, vol .06) and P3 (rating 1200, RD 100, vol .05), then P1's lists will read
             ['opponent_elo'] = [1400,1200], ['opponent_rd'] = [200,100] after 2 iterations of the for loop
             (assuming these matches occurred directly after each other in the dictionary"""
            ratings_dict[p1_id]['opponent_elo'].append(p2_elo)
            ratings_dict[p1_id]['opponent_deviation'].append(p2_deviation)
            ratings_dict[p2_id]['opponent_elo'].append(p1_elo)
            ratings_dict[p2_id]['opponent_deviation'].append(p1_deviation)

            # Appends the outcome to the correct player depending on who won the match
            ratings_dict[p1_id]['outcome'].append(v['p1_score'])
            ratings_dict[p2_id]['outcome'].append(v['p2_score'])

            # For computational efficiency, players are only updated if they are active (set here if they play a match)
            ratings_dict[p1_id]['active'] = 1
            ratings_dict[p2_id]['active'] = 1

            # Adds each player's elo, RD, and vol (from the end of the previous period) to each match in the
            # main match dictionary
            match_dict[k]['p1_glicko_rating'] = p1_elo
            match_dict[k]['p2_glicko_rating'] = p2_elo
            match_dict[k]['p1_glicko_deviation'] = p1_deviation
            match_dict[k]['p2_glicko_deviation'] = p2_deviation
            match_dict[k]['p1_glicko_volatility'] = glicko_dict[p1_id].vol
            match_dict[k]['p2_glicko_volatility'] = glicko_dict[p2_id].vol

        # Makes sure we only update players who are active (for computational efficiency)
        active_players_dict = {k: v for k, v in ratings_dict.items() if (v['active'] == 1)}

        # Loops through all active players to update their Glicko ratings
        for k, v in active_players_dict.items():
            # If they player has played 1 or more matches in this period, then update their Glicko stats
            if len(v['outcome']) > 0:
                glicko_dict[k].update_player(active_players_dict[k]['opponent_elo'], active_players_dict[k]['opponent_deviation'], active_players_dict[k]["outcome"])
            # If the player has not played a match in this period, then call did_not_compete()
            else:
                glicko_dict[k].did_not_compete()

            # Reset the appended lists for opponent data (to be re-populated with the next period's data)
            ratings_dict[k]['opponent_elo'] = []
            ratings_dict[k]['opponent_deviation'] = []
            ratings_dict[k]['outcome'] = []

        # Moves the period forward by the rating period
        rate_period_date = future_period_date
        loops_complete += 1
        print(f'Estimated time remaining: {((time.time()-start_time)*(num_loops-loops_complete)/(loops_complete)):.2f} seconds')

    #------------------------------------------------------------------
    # This section is purely for my own visualization, producing a DataFrame with sorted Glicko rankings at the
    # end of the data
    if show_rankings > 0:
        df['glicko_rating'] = 0
        counter = 0

        # Pulls the final Glicko ratings for each unique player, stores them in DF
        for i in enumerate(df.copy().index):
            df.loc[counter, 'glicko_rating'] = glicko_dict[i[1]].rating
            df.loc[counter, 'glicko_RD'] = glicko_dict[i[1]].rd
            df.loc[counter, 'id'] = i[1]
            counter += 1

        # Drops the index, joins with match_df to produce player names
        # This .join is not perfect, but since it is only for internal purposes, it's good enough
        df = df.drop(columns=[0])
        df = df.join(match_df.copy()[['p1_id', 'p1_name']].set_index('p1_id', drop=True), on='id',
                     how='left').drop_duplicates(subset='id', keep='first').reset_index()
        df = df.sort_values(by='glicko_rating', ascending=False)
        df = df.set_index(0, drop=True)
        df['index'] = np.arange(len(df))
        df = df.set_index('index', drop=True)
        # Drop unnecessary columns
        df = df.drop(
            ['opponent_elo', 'opponent_deviation', 'outcome',
             'active'], axis=1)
        # Gets rid of inactive players (inactive players will have very large Glicko RD's (usually larger than 80))
        # This cutoff can be adjusted in .main()
        df = df[df['glicko_RD'] < rd_cutoff].reset_index(drop=True)

        # Prints the dataframe
        print(f'\nGlicko rankings as of {rate_period_date}')
        print(df.head(show_rankings))

    # Reformats the dictionary back into a DataFrame for further analysis. Returns said DataFrame
    output_df = pd.DataFrame.from_dict(match_dict, orient='index')

    return output_df

def duplicate_df(match_df):
    """this function is not currently in use. originally, it duplicated p1/p2 features to make the data symmetrical.
    now, split_df2 has this purpose"""
    print(f'Duplicating DataFrame with flipped scores (to balance neural network output)')
    match_df_1 = match_df.copy(deep=True)
    match_df_1['winner'] = 0
    match_df_2 = match_df.copy(deep=True)
    match_df_2['winner'] = 1

    # Moves the columns around for P2 winner dataframe. Moves all columns that start with P1 to the columns that start
    # with P2

    column_headers = match_df_1.columns.values
    p1_colnames = []
    for i in column_headers:
        if str(i)[0:3] == 'p1_':
            p1_colnames.append(i[3:])

    for arg in [0, 1]:
        list = ['p1_', 'p2_']
        list1 = ['p2_', 'p1_']
        for i in p1_colnames:
            match_df_2.loc[:, f'{list[arg]}{i}'] = match_df_1.loc[:, f'{list1[arg]}{i}']

    combined_df = []
    combined_df.append(match_df_1)
    combined_df.append(match_df_2)
    match_df = pd.concat(combined_df, axis=0, ignore_index=True).sort_values(by='tourney_date',ascending=True)
    # print(match_df[match_df['tourney_name'] == 'Wroclaw CH'])
    return match_df


def date_features(match_df):
    """this function adds date features (year, sin(day) and cos(day)) to the dataset. sin and cos are used for the day
    features to account for seasonality. Dec 31 and Jan 1 should be considered relatively close, but if the data were
    modeled as a linear function, Jan 1 (1) and Dec 31 (365) would be interpreted as far apart by the NN"""
    print(f'\nAdded features for year and day of year (as sine/cosine)\n')
    match_df['tourney_date'] = pd.to_datetime(match_df['tourney_date'])
    match_df['year'] = match_df['tourney_date'].dt.strftime('%Y').astype(int)
    match_df['day_sin'] = match_df['tourney_date'].dt.strftime('%j').astype(int)
    match_df['day_sin'] = [math.sin(x*2*math.pi/365) for x in match_df['day_sin']]
    match_df['day_cos'] = match_df['tourney_date'].dt.strftime('%j').astype(int)
    match_df['day_cos'] = [math.cos(x*2*math.pi/365) for x in match_df['day_cos']]

    return match_df

def tourney_level(match_df):
    """this function consolidates the many tournament levels into 3: ATP, C, and S. it adds this level as a new
    column and returns the modified dataframe"""
    match_df['tourney_level_consolidated'] = match_df['tourney_level']
    match_df['tourney_level_consolidated'] = match_df['tourney_level_consolidated'].replace('G','ATP').replace('M','ATP').replace('A','ATP').replace('D','ATP').replace('F','ATP').replace('15','S').replace('25','S')
    print(f'Consolidated tournament levels into: {match_df.tourney_level_consolidated.unique()}.\n')
    return match_df

def player_wins(match_df, time_periods, rolling_windows):
    """this function calculates a rolling sum of each player's wins and losses over the specified time periods"""
    print('Adding player historical wins')
    df = match_df.copy(deep=True)
    levels = df.tourney_level_consolidated.unique()


    df['winner'] = 0
    # creates dummy variables for each tournament level. default is ATP, Q, and S
    levels_dummies = pd.get_dummies(df['tourney_level_consolidated'])
    df = df.join(levels_dummies)
    df = df.sort_values(['tourney_date','match_num'])
    df = df.set_index('tourney_date',drop=False)

    df['dt_index'] = df['tourney_date'] + df['match_num'].astype('timedelta64[s]')

    # creates a dataframe that adds players from both p1 and p2. this will store each player's match wins at all
    # points in time when they play
    p1_df = df[['p1_id','winner','dt_index','tourney_level_consolidated','tourney_name']].rename(columns={'p1_id':'player_id'})
    p1_df['winner'] = p1_df['winner'].replace(0,1)
    join_df = pd.concat([p1_df
                        ,df[['p2_id','winner','dt_index','tourney_level_consolidated','tourney_name']].rename(columns={'p2_id':'player_id'})]).set_index('dt_index').sort_index()
    level_names = []

    # creates new columns in the newly created dataframe for each level, and w/l (6 columns total). these will be used
    # to compute the rolling sum
    for level in levels:
        join_df[f'{level}_w'] = ((join_df['winner'] == 1) & (join_df['tourney_level_consolidated'] == level)).astype(int)
        join_df[f'{level}_l'] = ((join_df['winner'] == 0) & (join_df['tourney_level_consolidated'] == level)).astype(int)
        level_names.append(f'{level}_w')
        level_names.append(f'{level}_l')

    # groups the dataframe here for computational efficiency
    grouped_join_df = join_df.copy().groupby(['player_id'])

    # sets names to store column names; helps with renaming later
    player_colnames = []
    p1_colnames = []
    p2_colnames = []

    count = 0
    # iterates through all necessary combinations of level, result, and time periods and creates the rolling sum
    # within the newly created dataframe.
    for level in levels:
        for result in ['w','l']:
            for i in range(len(time_periods)):
                join_df[f'player_{result}_{level}_{time_periods[i]}'] = \
                    grouped_join_df[f'{level}_{result}'].transform(lambda x: x.rolling(rolling_windows[i], closed= 'left',).sum()).fillna(0)
                player_colnames.append(f'player_{result}_{level}_{time_periods[i]}')
                p1_colnames.append(f'p1_{result}_{level}_{time_periods[i]}')
                p2_colnames.append(f'p2_{result}_{level}_{time_periods[i]}')
                print(f'Percent complete: {100 * (count + 1) / (len(time_periods)*2*len(levels)):.2f}%')
                count += 1

    # drops unncessary columns and rows
    join_df = join_df.drop(['winner','tourney_level_consolidated'],axis=1)
    join_df = join_df.drop(level_names,axis=1)
    join_df = join_df.reset_index()

    # a few rows in the original dataset have duplicate entries. this line removes them
    join_df = join_df.drop_duplicates(subset=['player_id','dt_index','tourney_name'],keep='last')

    # pulls the correct information for p1/p2 wins from the newly created dataframe. indexes on dt_index
    df = df.merge(join_df, how='left', left_on= ['p1_id','dt_index','tourney_name'], right_on= ['player_id','dt_index','tourney_name'])
    df = df.drop('player_id',axis=1)
    df = df.rename(columns= dict(zip(player_colnames,p1_colnames)))

    df = df.merge(join_df, how='left', left_on= ['p2_id','dt_index','tourney_name'], right_on= ['player_id','dt_index','tourney_name'])
    df = df.drop('player_id',axis=1)
    df = df.rename(columns= dict(zip(player_colnames,p2_colnames)))

    return df

def tourney_history(match_df, time_periods, rolling_windows):
    # this section calculates historical player performance at the tournament in question, over the periods
    # last year, last 3 years, and career
    print('\nAdding player historical tournament performance...')
    df = match_df.copy(deep=True)
    # creating an index of each unique tournament. The current index has the unique tournament code, followed by the year.
    # this creates a new index which removes the year, creating an index of unique tournament ID's.
    df['tourney_code_no_year'] = df['tourney_id'].str.split(pat='-',expand=True,n=1)[1]

    """creates a de facto composite key for the dataset. In the unmodified dataset, the tourney_date column is the same
    for each match in the tournament, even if these matches occurred on different days. this would lead to data leakage
    if the individual matches were not ordered. the match_num column is the order of matches in each tournament, so
    combining the two columns allows us to correctly sort by date and time"""
    df['dt_index'] = df['tourney_date'] + df['match_num'].astype('timedelta64[s]')
    df = df.sort_values(['tourney_date','match_num'])
    df = df.set_index('dt_index',drop=True)

    # creates placeholder lists to make future transformations easier
    p1_list = ['p1_id','tourney_date','match_num','tourney_name']
    p2_list = ['p2_id','tourney_date','match_num','tourney_name']
    p1_elist = []
    p2_elist = []

    # to prevent redundancy and save time, dataframes are grouped ahead of time; these grouped df's are referenced later
    p1_grouped_df = df.groupby(['p1_id','tourney_code_no_year'])
    p2_grouped_df = df.groupby(['p2_id','tourney_code_no_year'])

    """this for loop iterates through all time periods passed into the function and creates a rolling sum of tournament
    wins and losses for each of those time periods. .rolling() is vectorized and provided significant performance
    improvements over previous versions of the code. due to the structure of the source data, only player 1 wins and
    player 2 losses can be aggregated here. player 2 wins and player 1 losses are then cross-referenced with these two
    columns (and the appropriate time index) after this for loop."""
    for i in range(len(time_periods)):
        df[f'p1_w_tourney_{time_periods[i]}'] = \
            p1_grouped_df['tourney_code_no_year'].transform(lambda x: x.shift(1).rolling(rolling_windows[i]).count()).astype(int)
        df[f'p2_l_tourney_{time_periods[i]}'] = \
            p2_grouped_df['tourney_code_no_year'].transform(lambda x: x.shift(1).rolling(rolling_windows[i]).count()).astype(int)
        # adding lists of features to make further transformations easier
        p2_list.append(f'p2_l_tourney_{time_periods[i]}')
        p1_list.append(f'p1_w_tourney_{time_periods[i]}')
        p2_elist.append(f'p2_l_tourney_{time_periods[i]}')
        p1_elist.append(f'p1_w_tourney_{time_periods[i]}')
        print(f'Percent complete: {100*(i+1) / len(time_periods):.2f}%')

    """at a high level, to calculate player 2 historical wins, this section searches backwards (on the dt_index)
    and finds the most recent instance where player 2's player_id appears in the player 1 column. It then pulls player
    1's wins from that row, and adds 1 to that total (because the player's ID appeared in the player 1 column, it can be
    assumed that this player won that match). the result is stored in the p2_wins column. the inverse is true for
    p1_losses """
    # this test_df column is what is searched backwards through.
    # TODO: evaluate data leakage on the backwards merge_asof(). fear is that by searching backwards, the gap between the 2 rows is added to the rolling window.
    merge_df = df[p2_list].copy()
    merge_df.loc[:, p2_elist] += 1
    merge_df = merge_df.rename(columns={
        'p2_l_tourney_l1y':'p1_l_tourney_l1y',
        'p2_l_tourney_l3y':'p1_l_tourney_l3y',
        'p2_l_tourney_career':'p1_l_tourney_career'
    })

    # sorting dataframe by index to ensure the merge_asof() functions appropriately. same thing as dt_index
    df = df.sort_values(['tourney_date','match_num'])
    merge_df = merge_df.sort_values(['tourney_date','match_num'])

    # computes the merge_asof(). searches backwards on the dt_index and merges the appropriate rows from merge_df
    df = pd.merge_asof(df, merge_df, left_by=['p1_id', 'tourney_name'], right_by=['p2_id', 'tourney_name'], on='dt_index', direction='backward', suffixes=['','_y'])
    df = df.drop(['p2_id_y','match_num_y','tourney_date_y'],axis=1)

    # resets the df's index, to confirm the future merge functions appropriately
    df['dt_index'] = df['tourney_date'] + df['match_num'].astype('timedelta64[s]')
    df = df.set_index('dt_index',drop=True)

    # same function as prior section, but computes for p2_wins instead of p1_losses
    merge_df = df[p1_list].copy()
    merge_df.loc[:, p1_elist] += 1
    merge_df = merge_df.rename(columns={
        'p1_w_tourney_l1y':'p2_w_tourney_l1y',
        'p1_w_tourney_l3y':'p2_w_tourney_l3y',
        'p1_w_tourney_career':'p2_w_tourney_career'
    })

    # sorting dataframe by index to ensure the merge_asof() functions appropriately. same thing as dt_index
    df = df.sort_values(['tourney_date','match_num'])
    merge_df = merge_df.sort_values(['tourney_date','match_num'])

    # computes the merge_asof(). searches backwards on the dt_index and merges the appropriate rows from merge_df
    df = pd.merge_asof(df, merge_df, left_by=['p2_id', 'tourney_name'], right_by=['p1_id', 'tourney_name'], on='dt_index', direction='backward', suffixes=['','_y'])
    df = df.drop(['p1_id_y','match_num_y','tourney_date_y'],axis=1)

    # loops through all columns that were just created, and fills all NAs (which occur at the first instance a player
    # appears at, due to .shift(1)). changes type from float to int
    for p in ['p1','p2']:
        for x in ['l','w']:
            for i in range(len(time_periods)):
                df[f'{p}_{x}_tourney_{time_periods[i]}'] = df[f'{p}_{x}_tourney_{time_periods[i]}'].fillna(0).astype(int)

    return df

def split_df_2(match_df):
    print('\nSplitting dataframe...')
    """currently, the winning player is always player 1. In this condition, the dataframe would be found to be biased
    by the neural network. Thus, we have to split the dataframe, allowing player 1 to be the winner half the time,
    and player 2 to be the winner half of the time."""

    # Creates a 'split' key column, defining whether player 0 or 1 will be the winner
    match_df['winner'] = np.random.randint(2, size=match_df.shape[0])

    # The next 4 lines are simply for user clarity. They display how many rows were assigned P1 and P2 as winners
    count_zero = match_df['winner'][match_df['winner'] == 0].count()
    count_one = match_df['winner'][match_df['winner'] == 1].count()
    print(f'setting {count_zero} matches to P1 winner.')
    print(f'setting {count_one} matches to P2 winner.')

    # Creates 2 dataframes to do modification on; one is all the rows that player 1 is the winner, and the other is
    # the dataframe where player 2 is the winner
    p1_winner = match_df[match_df['winner'] == 0]
    p2_winner = match_df[match_df['winner'] == 1]

    # Making a copy to prevent the Copy error
    p2_winner_copy = p2_winner.copy(deep=True)

    # Moves the columns around for P2 winner dataframe. Moves all columns that start with P1 to the columns that start
    # with P2
    p1_col_list = []
    p2_col_list = []
    col_list = []

    for col in match_df.columns:
        if str(col[:2]) == 'p1':
            col_list.append(str(col[2:]))

    player_list = ['p1','p2']

    # flips the appropriate feature columns in the new dataframe
    for col in col_list:
        p2_winner_copy.loc[:, f'{player_list[0]}{col}'] = p2_winner.loc[:, f'{player_list[1]}{col}']
        p2_winner_copy.loc[:, f'{player_list[1]}{col}'] = p2_winner.loc[:, f'{player_list[0]}{col}']


    # Creates an empty list, appends both of the split dataframes, then concatenates them
    combined_df = []
    combined_df.append(p1_winner)
    combined_df.append(p2_winner_copy)
    match_df = pd.concat(combined_df, axis=0).sort_index()

    return match_df

def time_on_court(match_df, games_base, year_base, time_periods, rolling_windows):
    """this section computes time on court for each player in each match through the time periods passed to the function.
    the defaults are last match, last 3 matches, and last 2 weeks"""
    df = match_df.copy(deep=True)

    # creates index and sorts by index, to account for removed matches in prior section
    df['dt_index'] = df['tourney_date'] + df['match_num'].astype('timedelta64[s]')
    df = df.set_index('dt_index',drop=True).sort_index()

    """one issue with this dataset is that many matches are missing match time. instead of removing these matches,
    I use increasingly general proxies to estimate the match time. first, the code attempts to fill the empty value
    with the average game time of all games within the last decade that have finished in the same number of games
    (rounded to 3 games). any values that are not filled here are then filled after removing the criteria that the
    averaged matches have to have occurred in the same decade (and before the match in question). any rows that are
     still unfilled are removed."""
    # creates a rounded_games column, to the nearest multiple of 3 (to group by later)
    df['total_games_rounded'] = df['total_games'].apply(lambda x: custom_round(x, games_base))
    df['decade'] = df['year'].apply(lambda x: custom_round(x, year_base))

    # fills the missing minutes rows with the median of past matches within the same decade, and with similar game count
    # TODO: check that these groupbys work correctly. check manually in Excel
    df['minutes'] = df.groupby(['total_games_rounded', 'decade'])['minutes'].transform(lambda x: x.fillna(x.expanding(1).median()))
    df['minutes'] = df.groupby('total_games_rounded')['minutes'].transform(lambda x: x.fillna(x.expanding(1).median()))
    print(f'\nRemoving {df["minutes"].isna().sum()} matches with no match time.')
    df = df.dropna(subset='minutes')
    print(f'Complete...\n')

    # sets the index for future sorting
    df['dt_index'] = df['tourney_date'] + df['match_num'].astype('timedelta64[s]')

    # creates a new dataframe that combines p1/p2 into one dataframe. this is for easier indexing and merging later
    player_df = pd.concat([df[['p1_id','minutes','dt_index','tourney_name']].rename(columns={'p1_id':'player_id'}),df[['p2_id','minutes','dt_index','tourney_name']].rename(columns={'p2_id':'player_id'})]).sort_index()
    player_df = player_df.drop_duplicates(subset=['player_id','dt_index','tourney_name'],keep='last')

    # groups the newly created player dataframe by player_id. to later aggregate on. grouped here for computational
    # efficiency
    grouped_player_df = player_df.copy().groupby('player_id')

    # loops through all time periods passed to the function and creates the rolling sum of match time (over the periods)
    print('Calculating player time on court...')
    for i in range(len(time_periods)):
        player_df[f'player_time_oncourt_{time_periods[i]}'] = \
            grouped_player_df['minutes'].transform(lambda x: x.rolling(rolling_windows[i], closed= 'left', min_periods=0).sum())
        print(f'Percent complete: {100*(i+1) / len(time_periods):.2f}%')

    # drops unnecessary columns from the newly created player dataframe
    player_df = player_df.drop(['dt_index','minutes'],axis=1)
    df = df.drop('dt_index',axis=1)

    # searches the newly created player dataframe and matches on the dt_index; pulls the match time from the dataframe
    for player in ['p1','p2']:
        df = df.merge(player_df, how='left', left_on= [f'{player}_id','dt_index','tourney_name'], right_on= ['player_id','dt_index','tourney_name'])
        df = df.drop('player_id',axis=1)
        for period in time_periods:
            df = df.rename(columns= {f'player_time_oncourt_{period}':f'{player}_time_oncourt_{period}'})

    return df

def custom_round(x, base):
    return int(base * round(float(x)/base))


def to_nn(match_df):
    """saves the final output to a csv"""
    match_df.to_csv(f'{wd}/Data/output_df3.csv')
    print(f'CSV saved to {wd}/Data/output_df.csv')


def main():
    # reading data; storing in DataFrames
    atp_df, futures_df, qual_challenger_df, rankings_df, player_df = read_files()

    # combining the 3 dataframes for later analysis
    match_df = combine_dfs(atp_df, futures_df, qual_challenger_df)

    # removing walkovers and retirements.
    match_df = remove_walkovers(match_df)

    # NOT CURRENTLY USED
    # renames winner/loser to p1/p2, move features as necessary
    # match_df = split_df(match_df)

    match_df = rename_columns(match_df)
    # removing unneeded features
    match_df = remove_features(match_df)

    # pulls tournament names, removes the year, to create unique key for every tournament (non-year dependent)
    tourney_list, no_year_tourney_list = get_tournaments(match_df)

    # ACTIVATE THE BELOW FUNCTION TO RESET CITIES
    # tourney_list = pull_cities(no_year_tourney_list)

    # attaches tournaments to their appropriate countries, to compute home court advantage
    match_df = attach_tournaments(match_df, no_year_tourney_list)

    # change to one-hot encoding of bo5, p1/p2 hand,
    match_df = one_hot_bof5(match_df)

    # removes a section of the dataframe, for computational purposes. if computing full dataset, comment out these lines
    match_df = match_df[700000:]
    # match_df = match_df[match_df['tourney_level'] == 'ATP']

    # pull rankings and append to dataframe
    match_df = set_h2h(match_df, rankings_df)

    # calculates player scores to input into Glicko-2 ranking system
    winner_weight = .6
    sets_weight = .2
    games_weight = .2
    match_df = winner_points(match_df, winner_weight, sets_weight, games_weight)


    # using the following input parameters, calculates the Glicko-2 ratings of players in dataset
    initial_rating = 2300
    initial_deviation = 350
    initial_volatility = .08
    rating_period = 40
    show_rankings = 100
    rd_cutoff = 80
    match_df = gluck_gluck2(match_df, initial_rating, initial_deviation, initial_volatility, rating_period, show_rankings, rd_cutoff)

    # adds date features (year, sin(day), cos(day)) to the dataframe
    match_df = date_features(match_df)

    # consolidates all tournament levels into 'ATP', 'C', and 'S'
    match_df = tourney_level(match_df)

    # calculates a rolling sum of p1/p2 wins and losses at all levels in the following time periods
    time_periods = ['l6m', 'l1y', 'career']
    rolling_windows = ['182d', '365d', '30000d']
    match_df = player_wins(match_df, time_periods, rolling_windows)

    # calculates a rolling sum of p1/p2 wins and losses at their current tournament in the following time periods
    time_periods = ['l1y', 'l3y', 'career']
    rolling_windows = ['365d', '1095d', '30000d']
    match_df = tourney_history(match_df, time_periods, rolling_windows)

    # calculates a rolling sum of time on court for all players in the following time periods.
    # fills missing match time data with averages; rounds match_games to games_base and rounds year to year_base
    games_base = 3
    year_base = 10
    time_periods = ['last_match', 'last_3_matches', 'last_2_weeks']
    rolling_windows = [1, 3, '14d']
    match_df = time_on_court(match_df, games_base, year_base, time_periods, rolling_windows)

    # splits the dataframe, for half of the dataset, makes p1 the winner, and for the other half, makes p2 the winner
    # match_df = split_df_2(match_df)

    # NO LONGER USED
    # duplicates the dataframe to make the neural network symmetrical
    # match_df = duplicate_df(match_df)


    print(match_df.head(10))
    print(match_df.tail(10))

    # saves the dataframe as a csv, outputs necessary features to the neural network
    to_nn(match_df)

    # sound to notify on code completion
    winsound.Beep(500, 100)

main()

# __________
# to test time on court
# moya_df = match_df[(match_df['p1_name'] == 'Carlos Moya') | (match_df['p2_name'] == 'Carlos Moya')]
# print(moya_df.head(50))
# # print(moya_df[moya_df['tourney_code_no_year'] == '580'].head(50))
# print(moya_df.loc[:, ['p1_name', 'p2_name', 'minutes', 'p1_time_oncourt_last_match', 'p1_time_oncourt_last_3_matches', 'p1_time_oncourt_last_2_weeks', 'p2_time_oncourt_last_match', 'p2_time_oncourt_last_3_matches', 'p2_time_oncourt_last_2_weeks']].head(900))

# match_df_s = match_df[['S','p1_id','p2_id','p1_w_tourney_career','p1_l_tourney_career','p2_w_tourney_career','p2_tourney_S_career','tourney_name']]
# match_df_s = match_df_s[match_df_s['S'] == 1]
# match_df = match_df[['ATP','p1_id','p2_id','p1_w_tourney_career','p1_l_tourney_career','p2_w_tourney_career','p2_l_tourney_career','tourney_name']]
# # match_df = match_df[match_df['ATP'] == 1]
#
#
# print(match_df[(match_df['p1_id'] == 104925) | (match_df['p2_id'] == 104925)].loc[:,['p1_name','p2_name','tourney_level_consolidated','p1_w_S_career','p1_l_S_career','p1_w_C_career','p1_l_C_career','p1_w_ATP_career','p1_l_ATP_career','p2_w_S_career','p2_l_S_career','p2_w_C_career','p2_l_C_career','p2_w_ATP_career','p2_l_ATP_career']])
