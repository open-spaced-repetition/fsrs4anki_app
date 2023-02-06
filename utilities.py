from functools import partial
import datetime
from zipfile import ZipFile

import sqlite3
import time

import gradio as gr
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os
from datetime import timedelta, datetime
from pathlib import Path

import torch
from sklearn.utils import shuffle

from model import Collection, init_w, FSRS, WeightClipper, lineToTensor


# Extract the collection file or deck file to get the .anki21 database.


def extract(file, prefix):
    proj_dir = Path(f'projects/{prefix}_{file.orig_name.replace(".", "_").replace("@", "_")}')
    with ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(proj_dir)
        # print(f"Extracted {file.orig_name} successfully!")
    return proj_dir


def create_time_series_features(revlog_start_date, timezone, next_day_starts_at, proj_dir,
                                progress=gr.Progress(track_tqdm=True)):
    if os.path.isfile(proj_dir / "collection.anki21b"):
        os.remove(proj_dir / "collection.anki21b")
        raise Exception(
                "Please export the file with `support older Anki versions` if you use the latest version of Anki.")
    elif os.path.isfile(proj_dir / "collection.anki21"):
        con = sqlite3.connect(proj_dir / "collection.anki21")
    elif os.path.isfile(proj_dir / "collection.anki2"):
        con = sqlite3.connect(proj_dir / "collection.anki2")
    else:
        raise Exception("Collection not exist!")
    cur = con.cursor()
    res = cur.execute("SELECT * FROM revlog")
    revlog = res.fetchall()

    df = pd.DataFrame(revlog)
    df.columns = ['id', 'cid', 'usn', 'r', 'ivl',
                  'last_lvl', 'factor', 'time', 'type']
    df = df[(df['cid'] <= time.time() * 1000) &
            (df['id'] <= time.time() * 1000) &
            (df['r'] > 0) &
            (df['id'] >= time.mktime(datetime.strptime(revlog_start_date, "%Y-%m-%d").timetuple()) * 1000)].copy()
    df['create_date'] = pd.to_datetime(df['cid'] // 1000, unit='s')
    df['create_date'] = df['create_date'].dt.tz_localize(
            'UTC').dt.tz_convert(timezone)
    df['review_date'] = pd.to_datetime(df['id'] // 1000, unit='s')
    df['review_date'] = df['review_date'].dt.tz_localize(
            'UTC').dt.tz_convert(timezone)
    df.drop(df[df['review_date'].dt.year < 2006].index, inplace=True)
    df.sort_values(by=['cid', 'id'], inplace=True, ignore_index=True)
    type_sequence = np.array(df['type'])
    df.to_csv(proj_dir / "revlog.csv", index=False)
    # print("revlog.csv saved.")
    df = df[(df['type'] == 0) | (df['type'] == 1)].copy()
    df['real_days'] = df['review_date'] - timedelta(hours=next_day_starts_at)
    df['real_days'] = pd.DatetimeIndex(df['real_days'].dt.floor('D')).to_julian_date()
    df.drop_duplicates(['cid', 'real_days'], keep='first', inplace=True)
    df['delta_t'] = df.real_days.diff()
    df.dropna(inplace=True)
    df['delta_t'] = df['delta_t'].astype(dtype=int)
    df['i'] = 1
    df['r_history'] = ""
    df['t_history'] = ""
    col_idx = {key: i for i, key in enumerate(df.columns)}

    # code from https://github.com/L-M-Sherlock/anki_revlog_analysis/blob/main/revlog_analysis.py
    def get_feature(x):
        for idx, log in enumerate(x.itertuples()):
            if idx == 0:
                x.iloc[idx, col_idx['delta_t']] = 0
            if idx == x.shape[0] - 1:
                break
            x.iloc[idx + 1, col_idx['i']] = x.iloc[idx, col_idx['i']] + 1
            x.iloc[idx + 1, col_idx[
                't_history']] = f"{x.iloc[idx, col_idx['t_history']]},{x.iloc[idx, col_idx['delta_t']]}"
            x.iloc[idx + 1, col_idx['r_history']] = f"{x.iloc[idx, col_idx['r_history']]},{x.iloc[idx, col_idx['r']]}"
        return x

    tqdm.pandas(desc='Saving Trainset')
    df = df.groupby('cid', as_index=False).progress_apply(get_feature)
    df["t_history"] = df["t_history"].map(lambda x: x[1:] if len(x) > 1 else x)
    df["r_history"] = df["r_history"].map(lambda x: x[1:] if len(x) > 1 else x)
    df.to_csv(proj_dir / 'revlog_history.tsv', sep="\t", index=False)
    # print("Trainset saved.")

    def cal_retention(group: pd.DataFrame) -> pd.DataFrame:
        group['retention'] = round(group['r'].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x]).mean(), 4)
        group['total_cnt'] = group.shape[0]
        return group

    tqdm.pandas(desc='Calculating Retention')
    df = df.groupby(by=['r_history', 'delta_t']).progress_apply(cal_retention)
    # print("Retention calculated.")
    df = df.drop(columns=['id', 'cid', 'usn', 'ivl', 'last_lvl', 'factor', 'time', 'type', 'create_date', 'review_date',
                          'real_days', 'r', 't_history'])
    df.drop_duplicates(inplace=True)
    df = df[(df['retention'] < 1) & (df['retention'] > 0)]

    def cal_stability(group: pd.DataFrame) -> pd.DataFrame:
        if group['i'].values[0] > 1:
            r_ivl_cnt = sum(group['delta_t'] * group['retention'].map(np.log) * pow(group['total_cnt'], 2))
            ivl_ivl_cnt = sum(group['delta_t'].map(lambda x: x ** 2) * pow(group['total_cnt'], 2))
            group['stability'] = round(np.log(0.9) / (r_ivl_cnt / ivl_ivl_cnt), 1)
        else:
            group['stability'] = 0.0
        group['group_cnt'] = sum(group['total_cnt'])
        group['avg_retention'] = round(
                sum(group['retention'] * pow(group['total_cnt'], 2)) / sum(pow(group['total_cnt'], 2)), 3)
        group['avg_interval'] = round(
                sum(group['delta_t'] * pow(group['total_cnt'], 2)) / sum(pow(group['total_cnt'], 2)), 1)
        del group['total_cnt']
        del group['retention']
        del group['delta_t']
        return group

    tqdm.pandas(desc='Calculating Stability')
    df = df.groupby(by=['r_history']).progress_apply(cal_stability)
    # print("Stability calculated.")
    df.reset_index(drop=True, inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_values(by=['r_history'], inplace=True, ignore_index=True)

    df_out = pd.DataFrame()
    if df.shape[0] > 0:
        for idx in tqdm(df.index):
            item = df.loc[idx]
            index = df[(df['i'] == item['i'] + 1) & (df['r_history'].str.startswith(item['r_history']))].index
            df.loc[index, 'last_stability'] = item['stability']
        df['factor'] = round(df['stability'] / df['last_stability'], 2)
        df = df[(df['i'] >= 2) & (df['group_cnt'] >= 100)]
        df['last_recall'] = df['r_history'].map(lambda x: x[-1])
        df = df[df.groupby(['i', 'r_history'])['group_cnt'].transform(max) == df['group_cnt']]
        df.to_csv(proj_dir / 'stability_for_analysis.tsv', sep='\t', index=None)
        # print("1:again, 2:hard, 3:good, 4:easy\n")
        # print(df[df['r_history'].str.contains(r'^[1-4][^124]*$', regex=True)][
        #     ['r_history', 'avg_interval', 'avg_retention', 'stability', 'factor', 'group_cnt']].to_string(
        #         index=False))
        # print("Analysis saved!")

        df_out = df[df['r_history'].str.contains(r'^[1-4][^124]*$', regex=True)][
            ['r_history', 'avg_interval', 'avg_retention', 'stability', 'factor', 'group_cnt']]
    return type_sequence, df_out


def train_model(proj_dir, progress=gr.Progress(track_tqdm=True)):
    model = FSRS(init_w)

    clipper = WeightClipper()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    dataset = pd.read_csv(proj_dir / "revlog_history.tsv", sep='\t', index_col=None,
                          dtype={'r_history': str, 't_history': str})
    dataset = dataset[(dataset['i'] > 1) & (dataset['delta_t'] > 0) & (dataset['t_history'].str.count(',0') == 0)]

    tqdm.pandas(desc='Tensorizing Line')
    dataset['tensor'] = dataset.progress_apply(lambda x: lineToTensor(list(zip([x['t_history']], [x['r_history']]))[0]),
                                               axis=1)
    # print("Tensorized!")

    pre_train_set = dataset[dataset['i'] == 2]
    # pretrain
    epoch_len = len(pre_train_set)
    n_epoch = 1
    pbar = tqdm(desc="Pre-training", colour="red", total=epoch_len * n_epoch)

    for k in range(n_epoch):
        for i, (_, row) in enumerate(shuffle(pre_train_set, random_state=2022 + k).iterrows()):
            model.train()
            optimizer.zero_grad()
            output_t = [(model.zero, model.zero)]
            for input_t in row['tensor']:
                output_t.append(model(input_t, *output_t[-1]))
            loss = model.loss(output_t[-1][0], row['delta_t'],
                              {1: 0, 2: 1, 3: 1, 4: 1}[row['r']])
            if np.isnan(loss.data.item()):
                # Exception Case
                # print(row, output_t)
                raise Exception('error case')
            loss.backward()
            optimizer.step()
            model.apply(clipper)
            pbar.update()
    pbar.close()
    for name, param in model.named_parameters():
        # print(f"{name}: {list(map(lambda x: round(float(x), 4), param))}")

    train_set = dataset[dataset['i'] > 2]
    epoch_len = len(train_set)
    n_epoch = 1
    print_len = max(epoch_len * n_epoch // 10, 1)
    pbar = tqdm(desc="Training", total=epoch_len * n_epoch)

    for k in range(n_epoch):
        for i, (_, row) in enumerate(shuffle(train_set, random_state=2022 + k).iterrows()):
            model.train()
            optimizer.zero_grad()
            output_t = [(model.zero, model.zero)]
            for input_t in row['tensor']:
                output_t.append(model(input_t, *output_t[-1]))
            loss = model.loss(output_t[-1][0], row['delta_t'],
                              {1: 0, 2: 1, 3: 1, 4: 1}[row['r']])
            if np.isnan(loss.data.item()):
                # Exception Case
                # print(row, output_t)
                raise Exception('error case')
            loss.backward()
            for param in model.parameters():
                param.grad[:2] = torch.zeros(2)
            optimizer.step()
            model.apply(clipper)
            pbar.update()

            # if (k * epoch_len + i) % print_len == 0:
                # print(f"iteration: {k * epoch_len + i + 1}")
                # for name, param in model.named_parameters():
                    # print(f"{name}: {list(map(lambda x: round(float(x), 4), param))}")
    pbar.close()

    w = list(map(lambda x: round(float(x), 4), dict(model.named_parameters())['w'].data))

    # print("\nTraining finished!")
    return w, dataset


def process_personalized_collection(requestRetention, w):
    my_collection = Collection(w)
    rating_dict = {1: "again", 2: "hard", 3: "good", 4: "easy"}
    rating_markdown = []
    for first_rating in (1, 2, 3, 4):
        rating_markdown.append(f'## First Rating: {first_rating} ({rating_dict[first_rating]})')
        t_history = "0"
        d_history = "0"
        r_history = f"{first_rating}"  # the first rating of the new card
        # print("stability, difficulty, lapses")
        for i in range(10):
            states = my_collection.states(t_history, r_history)
            # print('{0:9.2f} {1:11.2f} {2:7.0f}'.format(
            # *list(map(lambda x: round(float(x), 4), states))))
            next_t = max(round(float(np.log(requestRetention) / np.log(0.9) * states[0])), 1)
            difficulty = round(float(states[1]), 1)
            t_history += f',{int(next_t)}'
            d_history += f',{difficulty}'
            r_history += f",3"
        rating_markdown.append(f"**rating history**: {r_history}")
        rating_markdown.append(f"**interval history**: {t_history}")
        rating_markdown.append(f"**difficulty history**: {d_history}\n")
    rating_markdown = '\n\n'.join(rating_markdown)
    return my_collection, rating_markdown


def log_loss(my_collection, row):
    states = my_collection.states(row['t_history'], row['r_history'])
    row['log_loss'] = float(my_collection.model.loss(states[0], row['delta_t'], {1: 0, 2: 1, 3: 1, 4: 1}[row['r']]))
    return row


def my_loss(dataset, w):
    my_collection = Collection(init_w)
    tqdm.pandas(desc='Calculating Loss before Training')
    dataset = dataset.progress_apply(partial(log_loss, my_collection), axis=1)
    # print(f"Loss before training: {dataset['log_loss'].mean():.4f}")
    loss_before = f"{dataset['log_loss'].mean():.4f}"
    my_collection = Collection(w)
    tqdm.pandas(desc='Calculating Loss After Training')
    dataset = dataset.progress_apply(partial(log_loss, my_collection), axis=1)
    # print(f"Loss after training: {dataset['log_loss'].mean():.4f}")
    loss_after = f"{dataset['log_loss'].mean():.4f}"
    return f"""
*Loss before training*: {loss_before}

*Loss after training*: {loss_after}
    """


def cleanup(proj_dir: Path, files):
    """
    Delete all files in prefix that dont have filenames in files
    :param proj_dir:
    :param files:
    :return:
    """
    for file in proj_dir.glob('*'):
        if file.name not in files:
            os.remove(file)

