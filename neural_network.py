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
import xgboost as xgb
from sklearn.metrics import accuracy_score


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
    x = self.dropout_input(x)

    x = torch.relu(self.fc7(x))
    # x = self.dropout(x)

    x = torch.sigmoid(self.fc8(x))
    return x



class Net1(nn.Module):
  def __init__(self,input_shape):
    super(Net1,self).__init__()
    self.fc1 = nn.Linear(input_shape, 128)
    self.fc8 = nn.Linear(128, 1)

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
    # match_df = pd.read_csv(f'{wd}/Data/output_df.csv',index_col=0, header=0, parse_dates=['tourney_date'])
    match_df = pd.read_parquet(f'{wd}/Data/output_df.parquet.gzip')
    match_df = match_df.set_index('dt_index', drop=True).sort_index()
    return match_df

def feature_extraction(match_df, test_year, test_end):

    # print(match_df.shape[0])
    match_df = match_df[match_df.tourney_date < pd.to_datetime(test_end, format='%Y-%m-%d')]

    # print(match_df.shape[0])
    # print(match_df.shape[1])

    train_size = match_df[match_df['tourney_date'] < pd.to_datetime(test_year, format='%Y-%m-%d')].shape[0] / match_df.shape[0]
    print(f'Train split: {train_size*100:.2f}%')
    print(f'Test split: {(1-train_size)*100:.2f}%')
    match_df = baseline_accuracy(match_df)
    match_df = match_df.drop(['index','level_0','tourney_id','tourney_name','surface',
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
                              'p2_entry'
                              ],axis=1)

    match_df = match_df.drop(['p1_seed', 'p1_ht', 'p1_age', 'p2_seed', 'p2_ht', 'p2_age', 'p1_rank',
           'p1_rank_points', 'p2_rank', 'p2_rank_points', 'p1_home',
           'p2_home', 'tourney_level_consolidated', 'tourney_code_no_year','total_games_rounded', 'decade', 'total_games'], axis=1)

    # print(match_df[match_df.isna().any(axis=1)])


    # match_df = match_df.drop(['p1_time_oncourt_last_match','p1_time_oncourt_last_3_matches','p1_time_oncourt_last_2_weeks',
    #                           'p2_time_oncourt_last_match','p2_time_oncourt_last_3_matches','p2_time_oncourt_last_2_weeks'],axis=1)
    #
    # match_df = match_df.drop(['p1_glicko_rating','p1_glicko_deviation','p1_glicko_volatility',
    #                           'p2_glicko_rating','p2_glicko_deviation','p2_glicko_volatility'],axis=1)

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
    #
    # match_df = match_df.drop(['p1_w_tourney_l1y',
    #        'p2_l_tourney_l1y', 'p1_w_tourney_l3y', 'p2_l_tourney_l3y',
    #        'p1_w_tourney_career', 'p2_l_tourney_career', 'p1_l_tourney_l1y',
    #        'p1_l_tourney_l3y', 'p1_l_tourney_career', 'p2_w_tourney_l1y',
    #        'p2_w_tourney_l3y', 'p2_w_tourney_career'],axis=1)

    # match_df = match_df.drop(['p1_win_streak', 'p1_loss_streak', 'p2_win_streak',
    #        'p2_loss_streak', 'Carpet', 'Clay', 'Grass', 'Hard', 'p1_grass_wins',
    #        'p1_grass_losses', 'p1_hard_wins', 'p1_hard_losses', 'p1_clay_wins',
    #        'p1_clay_losses', 'p1_carpet_wins', 'p1_carpet_losses', 'p2_grass_wins',
    #        'p2_grass_losses', 'p2_hard_wins', 'p2_hard_losses', 'p2_clay_wins',
    #        'p2_clay_losses', 'p2_carpet_wins', 'p2_carpet_losses'], axis=1)

    # match_df = match_df.drop(['best_of_3', 'p1_left_handed', 'p1_right_handed', 'p2_left_handed', 'p2_right_handed', 'p1_h2h_wins', 'p2_h2h_wins', 'day_sin',
    #    'day_cos', 'ATP', 'C', 'S', 'p1_days_inactive', 'p2_days_inactive'], axis=1)

    # match_df = match_df.drop(['p1_glicko_surface_rating',
    #    'p2_glicko_surface_rating', 'p1_glicko_surface_deviation',
    #    'p2_glicko_surface_deviation', 'p1_glicko_surface_volatility',
    #    'p2_glicko_surface_volatility'], axis=1)

    # print(match_df.shape[1])

    # for i in range(1,34):
    #     match_df = match_df.drop([f'p1_seed_{i}'], axis=1)
    #     match_df = match_df.drop([f'p2_seed_{i}'], axis=1)
    #
    # match_df = match_df.drop(['p1_seed_35', 'p2_seed_34', 'p2_seed_35'], axis=1)
    #
    # match_df = match_df.drop(['F', 'SF', 'QF', 'R128', 'R64', 'R32', 'R16', 'RR', 'BR', 'ER', 'Q1', 'Q2', 'Q3', 'Q4'], axis=1)




    print(match_df.columns.tolist())
    print(match_df.shape[1])

    return match_df, train_size

def train_test_split_df(match_df, train_percent, ATP_only):
    X = match_df.copy(deep=True).drop(['winner','baseline_prediction'],axis=1).reset_index(drop=True)
    y = match_df.copy(deep=True)['winner'].reset_index(drop=True)
    baseline_x = match_df.copy(deep=True).loc[:, ['baseline_prediction','year']].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_percent,shuffle=False, random_state=420)
    baseline_train, baseline_test = train_test_split(baseline_x, train_size=train_percent, shuffle=False, random_state=420)

    print(X_test.shape[0])

    if ATP_only == 1:
        mask = X_test['ATP'] > 0
        X_test = X_test[mask]
        baseline_test = baseline_test[mask]
        y_test = y_test[mask]


    print(X_test.shape[0])

    dtrain_reg = xgb.DMatrix(X_train, y_train)
    dtest_reg = xgb.DMatrix(X_test, y_test)


    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # print(y_train)

    return X_train, X_test, y_train, y_test, baseline_train, baseline_test, X, y, dtrain_reg, dtest_reg


def run_nn(X_train, X_test, y_train, y_test, baseline_train, baseline_test, X, y, num_epochs, batch_size, learning_rate, model, name):
    print(f'Learning rate: {learning_rate}')
    best_acc = 0
    trainset = dataset(X_train,y_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.8)
    loss_fn = nn.BCELoss()
    early_stopper = EarlyStopper(patience=7, minima_patience=5, min_delta=.001)

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
                train_correct += (predicted.reshape(-1).detach().numpy().round() == y_train.detach().numpy()).sum()
                train_size += len(y_train.detach().numpy())
            model.train()

        # accuracy
        train_acc = train_correct/train_size
        model.eval()
        with torch.no_grad():
            predicted = model(torch.tensor(X_test, dtype=torch.float32))
            test_acc = (predicted.reshape(-1).detach().numpy().round() == y_test).mean()
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = model
                best_model_index = i


        model.train()
        scheduler.step()


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

    print(f'Saving model to : {wd}/models/model_2_26_batch{batch_size}lr{learning_rate}_{test_accur[best_model_index]:.5f}.pth')
    torch.save(best_model.state_dict(), f'{wd}/models/model_{name}layers_batch{batch_size}lr{learning_rate}_{test_accur[best_model_index]:.5f}.pth')
    return best_acc


# def lin_reg(X_train, X_test, y_train, y_test, baseline_train, baseline_test, X, y):

def xgboost(train, test, params, n, y_test, early_stopping):

    evals = [(train, "train"), (test, "validation")]

    # print(early_stopping)

    model = xgb.train(
        params = params,
        dtrain=train,
        num_boost_round=n,
        evals=evals,
        verbose_eval=50,
        early_stopping_rounds=early_stopping
    )
    preds = model.predict(test).round()

    feature_scores = (model.get_score(importance_type='gain'))
    # print(feature_scores)
    importances = pd.DataFrame.from_dict(feature_scores, orient='index')
    # print(importances)
    importances = importances.rename(columns={0: 'importance'})
    importances = importances.sort_values(by=['importance'], ascending=False)
    print(importances.head(20))

    acc = accuracy_score(y_test, preds)

    print(f'Accuracy of XGBoost: {acc:.5f}')

    results = xgb.cv(

        params, train,

        num_boost_round=n,

        nfold=5,

        early_stopping_rounds=20

    )

    print(results.head())
    best_loss = results['test-logloss-mean'].min()
    print(f'5f-cv loss: {best_loss:.5f}')


def main():
    match_df = import_data()

    test_cutoff = '2022-01-01'
    test_end = '2023-01-01'
    match_df, train_percent = feature_extraction(match_df, test_cutoff, test_end)
    # match_df = match_df[790000:]
    # print(match_df.tail(100))

    ATP_only = 1
    X_train, X_test, y_train, y_test, baseline_train, baseline_test, X, y, dtrain_reg, dtest_reg = train_test_split_df(match_df, train_percent, ATP_only)

    params = {"objective": "binary:logistic", "tree_method": "gpu_hist"}
    n=1000
    early_stopping_rounds = 50
    xgboost(dtrain_reg, dtest_reg, params, n, y_test, early_stopping_rounds)

    # num_epochs = 100
    # batch_size = 128
    # learning_rate = .001
    # nn_model = Net3(input_shape=X.shape[1])
    # run_nn(X_train, X_test, y_train, y_test, baseline_train, baseline_test, X, y, num_epochs, batch_size, learning_rate, nn_model)

    num_epochs = 100
    batch_size = 64
    torch.manual_seed(0)
    # learning_rate = .1
    learning_rates = [.0008]
    accs = []
    name = '4'
    input_p = .2
    ps = [.5]
    for lr in learning_rates:
        for p in ps:
            print(f'Dropout rate: {p}')
            nn_model = Net4(input_shape=X.shape[1], input_p=input_p, p=p)
            best_acc = run_nn(X_train, X_test, y_train, y_test, baseline_train, baseline_test, X, y, num_epochs, batch_size, lr, nn_model, name)
            accs.append(best_acc)
    winsound.Beep(500, 100)
    print(learning_rates)
    print(accs)





main()