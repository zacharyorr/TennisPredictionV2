import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
import winsound
# import xgboost as xgb
import xgboost.sklearn as xgb
from sklearn.metrics import accuracy_score
import copy


wd = 'C:/Users/zacha/PycharmProjects/TennisPredictionV2'
os.chdir(wd)
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_rows = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx],self.y[idx]

class EarlyStopper:
    def __init__(self, patience, minima_patience, min_delta):
        self.patience = patience
        self.minima_patience = minima_patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_acc = np.NINF
        self.max_train_acc = np.NINF
        self.minima_counter = 0

    def early_stop(self, validation_acc):
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.counter = 0
        elif validation_acc < (self.max_validation_acc - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    def local_min(self, train_acc):
        if train_acc != self.max_train_acc:
            self.max_train_acc = train_acc
            self.minima_counter = 0
        elif train_acc == self.max_train_acc:
            self.minima_counter += 1
            if self.minima_counter >= self.minima_patience:
                return True
        return False

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
    # x = self.dropout(x)

    x = torch.relu(self.fc2(x))
    # x = self.dropout(x)

    x = torch.relu(self.fc3(x))
    # x = self.dropout(x)

    x = torch.relu(self.fc4(x))
    # x = self.dropout(x)

    x = torch.relu(self.fc5(x))
    # x = self.dropout(x)

    x = torch.relu(self.fc6(x))
    # x = self.dropout(x)

    x = torch.relu(self.fc7(x))
    # x = self.dropout(x)

    x = torch.sigmoid(self.fc8(x))
    return x

class Net5(nn.Module):
  def __init__(self, input_shape, input_p, p):
      super(Net5, self).__init__()
      self.fc1 = nn.Linear(input_shape, 512)

      self.fc8 = nn.Linear(512, 1)
      self.dropout_input = nn.Dropout(input_p)
      self.dropout = nn.Dropout(p)

  def forward(self, x):
      x = torch.relu(self.fc1(x))
      # x = self.dropout(x)

      x = torch.sigmoid(self.fc8(x))
      return x



class Net1(nn.Module):
  def __init__(self,input_shape):
    super(Net1,self).__init__()
    self.fc1 = nn.Linear(input_shape, 512)
    self.fc8 = nn.Linear(512, 1)

  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.sigmoid(self.fc8(x))
    return x

def baseline_accuracy(df):
    df = df.copy(deep=True)
    df['baseline_prediction'] = ((df['p1_rank'] - df['p2_rank']) > 0).astype(int)
    return df


def import_data():

    # import data
    match_df = pd.read_parquet(f'{wd}/Data/output_df.parquet.gzip')
    match_df = match_df.set_index('dt_index', drop=True).sort_index()
    return match_df

def feature_extraction(match_df, test_year, test_end):

    match_df = match_df[match_df.tourney_date < pd.to_datetime(test_end, format='%Y-%m-%d')]

    print(match_df.head(10))

    train_size = match_df[match_df['tourney_date'] < pd.to_datetime(test_year, format='%Y-%m-%d')].shape[0] / match_df.shape[0]
    print(f'Train split: {train_size*100:.2f}%')
    print(f'Test split: {(1-train_size)*100:.2f}%')
    match_df = baseline_accuracy(match_df)
    match_df = match_df.drop(['index','level_0','tourney_id','tourney_name','surface',
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
                              'country_code',
                              'p1_score',
                              'p2_score',
                              'p1_entry',
                              'p2_entry',
                              'country_code',
                              'city_name'
                              ],axis=1)

    match_df = match_df.drop(['p1_seed', 'p1_ht', 'p2_seed', 'p2_ht', 'p1_rank',
           'p1_rank_points', 'p2_rank', 'p2_rank_points', 'tourney_level_consolidated', 'tourney_code_no_year','total_games_rounded', 'decade', 'total_games'], axis=1)

    match_df = match_df.drop(['BR', 'ER', 'F', 'Q1', 'Q2', 'Q3', 'Q4', 'QF', 'R128', 'R16', 'R32', 'R64', 'RR', 'SF'], axis=1)

    # match_df = match_df.drop(['p1_time_oncourt_last_match','p1_time_oncourt_last_3_matches','p1_time_oncourt_last_2_weeks',
    #                           'p2_time_oncourt_last_match','p2_time_oncourt_last_3_matches','p2_time_oncourt_last_2_weeks'],axis=1)

    # match_df = match_df.drop(['p1_glicko_rating','p1_glicko_deviation',
    #                           'p2_glicko_rating','p2_glicko_deviation'],axis=1)

    # match_df = match_df.drop(['p1_glicko_volatility','p2_glicko_volatility'],axis=1)


    # match_df = match_df.drop(['p1_w_S_l6m', 'p1_w_S_l1y', 'p1_w_S_career',
    #        'p1_l_S_l6m', 'p1_l_S_l1y', 'p1_l_S_career', 'p1_w_C_l6m', 'p1_w_C_l1y',
    #        'p1_w_C_career', 'p1_l_C_l6m', 'p1_l_C_l1y', 'p1_l_C_career',
    #        'p1_w_ATP_l6m', 'p1_w_ATP_l1y', 'p1_w_ATP_career', 'p1_l_ATP_l6m',
    #        'p1_l_ATP_l1y', 'p1_l_ATP_career', 'p2_w_S_l6m', 'p2_w_S_l1y',
    #        'p2_w_S_career', 'p2_l_S_l6m', 'p2_l_S_l1y', 'p2_l_S_career',
    #        'p2_w_C_l6m', 'p2_w_C_l1y', 'p2_w_C_career', 'p2_l_C_l6m', 'p2_l_C_l1y',
    #        'p2_l_C_career', 'p2_w_ATP_l6m', 'p2_w_ATP_l1y', 'p2_w_ATP_career',
    #        'p2_l_ATP_l6m', 'p2_l_ATP_l1y', 'p2_l_ATP_career'],axis=1)

    # match_df = match_df.drop(['p1_h2h_wins', 'p2_h2h_wins'],axis=1)

    # match_df = match_df.drop(['p1_w_tourney_l6m',
    #        'p2_l_tourney_l6m', 'p1_w_tourney_l3y', 'p2_l_tourney_l3y',
    #        'p1_w_tourney_career', 'p2_l_tourney_career', 'p1_l_tourney_l6m',
    #        'p1_l_tourney_l3y', 'p1_l_tourney_career', 'p2_w_tourney_l6m',
    #        'p2_w_tourney_l3y', 'p2_w_tourney_career'],axis=1)

    # match_df = match_df.drop(['p1_win_streak', 'p1_loss_streak', 'p2_win_streak',
    #        'p2_loss_streak'], axis=1)

    # match_df = match_df.drop(['Carpet', 'Clay', 'Grass', 'Hard', 'p1_grass_wins',
    #        'p1_grass_losses', 'p1_hard_wins', 'p1_hard_losses', 'p1_clay_wins',
    #        'p1_clay_losses', 'p1_carpet_wins', 'p1_carpet_losses', 'p2_grass_wins',
    #        'p2_grass_losses', 'p2_hard_wins', 'p2_hard_losses', 'p2_clay_wins',
    #        'p2_clay_losses', 'p2_carpet_wins', 'p2_carpet_losses'], axis=1)

    # match_df = match_df.drop(['best_of_3', 'p1_left_handed', 'p1_right_handed', 'p2_left_handed', 'p2_right_handed', 'day_sin',
    #    'day_cos', 'C', 'S', 'p1_days_inactive', 'p2_days_inactive'], axis=1)

    # match_df = match_df.drop(['p1_glicko_surface_rating',
    #    'p2_glicko_surface_rating', 'p1_glicko_surface_deviation',
    #    'p2_glicko_surface_deviation', 'p1_glicko_surface_volatility',
    #    'p2_glicko_surface_volatility'], axis=1)



    for i in range(1,34):
        match_df = match_df.drop([f'p1_seed_{i}'], axis=1)
        match_df = match_df.drop([f'p2_seed_{i}'], axis=1)

    match_df = match_df.drop(['p1_seed_35', 'p2_seed_34', 'p2_seed_35'], axis=1)

    # match_df = match_df.drop(['tournament_search', 'city'], axis=1)
    #
    # match_df = match_df.drop(['F', 'SF', 'QF', 'R128', 'R64', 'R32', 'R16', 'RR', 'BR', 'ER', 'Q1', 'Q2', 'Q3', 'Q4'], axis=1)


    return match_df, train_size

def train_test_split_df(match_df, train_percent, ATP_only):
    X = match_df.copy(deep=True).drop(['winner','baseline_prediction'],axis=1).reset_index(drop=True)
    y = match_df.copy(deep=True)['winner'].reset_index(drop=True)
    baseline_x = match_df.copy(deep=True).loc[:, ['baseline_prediction','year']].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_percent,shuffle=False, random_state=420)
    baseline_train, baseline_test = train_test_split(baseline_x, train_size=train_percent, shuffle=False, random_state=420)


    if ATP_only == 1:
        mask = X_test['ATP'] > 0
        X_test = X_test[mask]
        baseline_test = baseline_test[mask]
        y_test = y_test[mask]

    dtrain_reg = X
    dtest_reg = y

    X = X.drop(['tourney_date'], axis=1)
    X_train = X_train.drop(['tourney_date'], axis=1)
    X_test = X_test.drop(['tourney_date'], axis=1)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test, baseline_train, baseline_test, X, y, dtrain_reg, dtest_reg



def train_test_split_df_months(match_df, train_percent, ATP_only, max_date, max_cutoff, test_cutoff, iterations, num_epochs, batch_size, lr, nn_model, name, momentum, nesterov, patience):
    X = match_df.copy(deep=True).drop(['winner','baseline_prediction'],axis=1).reset_index(drop=True)
    y = match_df.copy(deep=True)['winner'].reset_index(drop=True)
    baseline_x = match_df.copy(deep=True).loc[:, ['baseline_prediction','year']].reset_index(drop=True)

    combined_preds = []
    accs = []

    test_end = min(max_date, pd.to_datetime(max_cutoff))
    print(test_end)

    num_months = 12 * (test_end.year - pd.to_datetime(test_cutoff).year) + test_end.month - pd.to_datetime(test_cutoff).month + 1
    # print(f'num months: {num_months}')
    print(f'Number of periods: {num_months}')

    cutoff_date = pd.to_datetime(test_cutoff)

    backup_nn_model = copy.deepcopy(nn_model)

    for i in range(num_months):

        nn_model = copy.deepcopy(backup_nn_model)

        if iterations == 'months':
            next_cutoff_date = (pd.to_datetime(cutoff_date) + np.timedelta64(31, 'D')).replace(day=1)
        if iterations == 'weeks':
            next_cutoff_date = (pd.to_datetime(cutoff_date) + np.timedelta64(7, 'D')).replace(day=1)
        if iterations == 'years':
            next_cutoff_date = (pd.to_datetime(cutoff_date) + np.timedelta64(1, 'Y')).replace(day=1)

        X_train = X[X['tourney_date'] < cutoff_date]
        y_train = y[X['tourney_date'] < cutoff_date]

        X_test = X[(X['tourney_date'] >= cutoff_date)]
        y_test = y[(X['tourney_date'] >= cutoff_date)]

        preds_len = len(X[(X['tourney_date'] >= cutoff_date) & (X['tourney_date'] < next_cutoff_date) & (X['ATP'] == 1)])

        baseline_test = baseline_x[(X['tourney_date'] >= cutoff_date)]


        X_test = X_test.drop(['tourney_date'], axis=1)
        X_train = X_train.drop(['tourney_date'], axis=1)



        if ATP_only == 1:
            mask = X_test['ATP'] > 0
            X_test = X_test[mask]
            baseline_test = baseline_test[mask]
            y_test = y_test[mask]


        if len(X_test) > 0:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            best_acc, preds = run_nn(X_train, X_test, y_train, y_test, baseline_test, X, y, num_epochs, batch_size, lr, nn_model, name, momentum, nesterov, cutoff_date, next_cutoff_date, patience)

            accs.append(best_acc)
            for pred in preds[:preds_len]:
                combined_preds.append(pred)

        cutoff_date = next_cutoff_date
        print(f'\n')

    acc = (y[(X['tourney_date'] >= pd.to_datetime(test_cutoff)) & (X['ATP'] == 1)] == combined_preds).mean()

    print(f'Accuracy of monthly neural network: {acc:.5f}')




def run_nn(X_train, X_test, y_train, y_test, baseline_test, X, y, num_epochs, batch_size, learning_rate, model, name, m, n, period_start, period_end, patience):
    print(f'Learning rate: {learning_rate}')
    best_acc = 0
    trainset = dataset(X_train,y_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=m, nesterov=n)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.8)
    loss_fn = nn.BCELoss()
    early_stopper = EarlyStopper(patience=patience, minima_patience=5, min_delta=.001)

    losses = []
    train_accur = []
    test_accur = []

    start_time = time.time()

    baseline_accuracy = (baseline_test['baseline_prediction'] == y_test).mean()
    print(f'Baseline accuracy: {baseline_accuracy}')

    for i in range(num_epochs):
        train_correct = 0
        train_size = 0

        for j, (X_train, y_train) in enumerate(trainloader):
            # print(device)

            X_test = torch.tensor(X_test, dtype=torch.float32)

            if torch.cuda.is_available():
                # print('hi')
                X_train = X_train.to('cuda')
                y_train = y_train.to('cuda')
                X_test = X_test.to('cuda')

            # calculate output
            output = model(X_train)
            # print(output)

            # calculate loss
            loss = loss_fn(output, y_train.reshape(-1, 1))

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                predicted = model(torch.tensor(X_train, dtype=torch.float32))
                train_correct += (predicted.cpu().reshape(-1).detach().numpy().round() == y_train.cpu().detach().numpy()).sum()
                train_size += len(y_train.cpu().detach().numpy())
            model.train()

        # accuracy
        train_acc = train_correct/train_size
        model.eval()
        with torch.no_grad():
            predicted = model(X_test)
            test_acc = (predicted.cpu().reshape(-1).detach().numpy().round() == y_test).mean()
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = copy.deepcopy(model)
                best_model_index = i


        model.train()
        # scheduler.step()


        losses.append(loss)
        train_accur.append(train_acc)
        test_accur.append(test_acc)
        print("epoch {}\tloss : {}\t train accuracy : {}\t test accuracy : {}".format((i+1), loss, train_acc, test_acc))
        print(f'Time elapsed: {time.time()-start_time}')
        print(f'Estimated time remaining: {((time.time() - start_time) * (num_epochs - i+1) / (i+1)):.2f} seconds')

        if early_stopper.early_stop(test_acc):
            print(f'Early stopping after {i} epochs...')
            break
        if early_stopper.local_min(train_acc):
            print(f'Local minima... Early stopping after {i} epochs...')
            break

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot([tensor.item() for tensor in losses], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Train Accuracy', color=color)
    ax2.plot(train_accur, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    color = 'tab:green'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(test_accur, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    print(f'Saving loss figure to: {wd}/images/model_2_26_batch{batch_size}lr{learning_rate}_{test_accur[best_model_index]:.5f}.png')
    fig.savefig(f'{wd}/images/model_2_26_batch{batch_size}lr{learning_rate}_{test_accur[best_model_index]:.5f}.png')
    plt.clf()

    print(f'Saving model to : {wd}/models/model_{period_start.year}{period_start.month}_{period_end.year}{period_end.month}.pth')
    torch.save(best_model.state_dict(), f'{wd}/models/model_{learning_rate}_online_{period_start.year}{period_start.month}_{period_end.year}{period_end.month}.pth')
    model.eval()
    with torch.no_grad():
        predicted = best_model(X_test)
        preds = (predicted.cpu().reshape(-1).detach().numpy().round())


    # print(best_acc)
    # print(type(best_acc))

    return best_acc, preds


def xgboost(X, y, params, n, y_test_total, early_stopping, train_percent, iterations, ATP_only, test_cutoff, max_date, max_cutoff):

    preds = []

    test_end = min(max_date, pd.to_datetime(max_cutoff))
    print(test_end)

    num_months = 12 * (test_end.year - pd.to_datetime(test_cutoff).year) + test_end.month - pd.to_datetime(test_cutoff).month + 1
    # print(f'num months: {num_months}')

    cutoff_date = pd.to_datetime(test_cutoff)

    for i in range(num_months):

        if iterations == 'm':
            next_cutoff_date = (pd.to_datetime(cutoff_date) + np.timedelta64(31, 'D')).replace(day=1)
        if iterations == 'w':
            next_cutoff_date = (pd.to_datetime(cutoff_date) + np.timedelta64(7, 'D')).replace(day=1)
        if iterations == 'y':
            next_cutoff_date = (pd.to_datetime(cutoff_date) + np.timedelta64(1, 'Y')).replace(day=1)

        # print(f'Cutoff date {cutoff_date}')
        # print(f'Cutoff date {next_cutoff_date}')


        X_train = X[X['tourney_date'] < cutoff_date]
        y_train = y[X['tourney_date'] < cutoff_date]

        X_test = X[(X['tourney_date'] >= cutoff_date) & (X['tourney_date'] < next_cutoff_date)]
        y_test = y[(X['tourney_date'] >= cutoff_date) & (X['tourney_date'] < next_cutoff_date)]

        X_test = X_test.drop(['tourney_date'], axis=1)
        X_train = X_train.drop(['tourney_date'], axis=1)
        if ATP_only == 1:
            mask = X_test['ATP'] > 0
            X_test = X_test[mask]
            y_test = y_test[mask]

        train = xgb.DMatrix(X_train, y_train)
        test = xgb.DMatrix(X_test, y_test)

        evals = [(train, "train"), (test, "validation")]

        if X_test.shape[0] > 0:
            model = xgb.train(
                params = params,
                dtrain=train,
                num_boost_round=n,
                evals=evals,
                verbose_eval=False,
                early_stopping_rounds=early_stopping
            )

            model.save_model(f'xgboost/xgboost_iteration_{cutoff_date.year}{cutoff_date.month}_{next_cutoff_date.year}{next_cutoff_date.month}.json')
            for pred in model.predict(test, ntree_limit=model.best_ntree_limit).round():
                preds.append(pred)
            # print(len(preds))

        cutoff_date = next_cutoff_date

    feature_scores = (model.get_score(importance_type='gain'))
    importances = pd.DataFrame.from_dict(feature_scores, orient='index')
    importances = importances.rename(columns={0: 'importance'})
    importances = importances.sort_values(by=['importance'], ascending=False)
    # print(importances)


    acc = accuracy_score(y[(X['tourney_date'] >= pd.to_datetime(test_cutoff)) & (X['ATP'] == 1)], preds)

    print(f'Accuracy of XGBoost: {acc:.5f}')



def main():
    match_df = import_data()

    test_cutoff = '2015-01-01'
    test_end = '2019-01-01'
    max_date = match_df['tourney_date'].max()
    print(max_date)
    match_df, train_percent = feature_extraction(match_df, test_cutoff, test_end)
    # match_df = match_df[650000:]
    # print(match_df.tail(100))

    ATP_only = 1
    X_train, X_test, y_train, y_test, baseline_train, baseline_test, X, y, dtrain_reg, dtest_reg = train_test_split_df(match_df, train_percent, ATP_only)

    params = {"objective": "binary:logistic", "tree_method": "gpu_hist"}
    n=1000
    early_stopping_rounds = 50
    iterations = 'months'
    # xgboost(dtrain_reg, dtest_reg, params, n, y_test, early_stopping_rounds, train_percent, iterations, ATP_only, test_cutoff, max_date, test_end)

    num_epochs = 100
    batch_size = 64
    torch.manual_seed(0)
    # learning_rate = .1
    learning_rates = [.00025, .0002, .0001]
    accs = []
    name = '5'
    input_p = 0
    ps = [0, .3, .4, .5, .6, .7]
    momentum = .9
    nesterov = True
    patience = 20
    print(f'xshape: {X.shape[1]}')
    nn_model = Net4(input_shape=X.shape[1], input_p=input_p, p=ps[0]).to(device)
    torch.cuda.synchronize()

    if iterations == 'months':
        train_test_split_df_months(match_df, train_percent, ATP_only, max_date, test_end, test_cutoff, iterations, num_epochs, batch_size, learning_rates[0], nn_model, name, momentum, nesterov, patience)

    nn_model = Net4(input_shape=X.shape[1], input_p=input_p, p=ps[0]).to(device)
    torch.cuda.synchronize()

    if iterations == 'months':
        train_test_split_df_months(match_df, train_percent, ATP_only, max_date, test_end, test_cutoff, iterations, num_epochs, batch_size, learning_rates[1], nn_model, name, momentum, nesterov, patience)

    nn_model = Net4(input_shape=X.shape[1], input_p=input_p, p=ps[0]).to(device)
    torch.cuda.synchronize()

    if iterations == 'months':
        train_test_split_df_months(match_df, train_percent, ATP_only, max_date, test_end, test_cutoff, iterations, num_epochs, batch_size, learning_rates[2], nn_model, name, momentum, nesterov, patience)

    nn_model = Net4(input_shape=X.shape[1], input_p=input_p, p=ps[0]).to(device)
    torch.cuda.synchronize()

    if iterations == 'months':
        train_test_split_df_months(match_df, train_percent, ATP_only, max_date, test_end, test_cutoff, iterations, num_epochs, batch_size, learning_rates[3], nn_model, name, momentum, nesterov, patience)

    nn_model = Net4(input_shape=X.shape[1], input_p=input_p, p=ps[0]).to(device)
    torch.cuda.synchronize()

    if iterations == 'months':
        train_test_split_df_months(match_df, train_percent, ATP_only, max_date, test_end, test_cutoff, iterations, num_epochs, batch_size, learning_rates[4], nn_model, name, momentum, nesterov, patience)

    nn_model = Net4(input_shape=X.shape[1], input_p=input_p, p=ps[0]).to(device)
    torch.cuda.synchronize()

    if iterations == 'months':
        train_test_split_df_months(match_df, train_percent, ATP_only, max_date, test_end, test_cutoff, iterations, num_epochs, batch_size, learning_rates[5], nn_model, name, momentum, nesterov, patience)

















    #
    # for lr in learning_rates:
    #     for p in ps:
    #         print(f'Dropout rate: {p}')
    #         nn_model = Net4(input_shape=X.shape[1], input_p=input_p, p=p)
    #         best_acc = run_nn(X_train, X_test, y_train, y_test, baseline_test, X, y, num_epochs, batch_size, lr, nn_model, name, momentum, nesterov, patience)
    #         accs.append(best_acc)
    winsound.Beep(500, 100)
    # print(learning_rates)
    # print(accs)
    #




main()