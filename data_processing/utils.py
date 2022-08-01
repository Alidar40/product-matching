import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from torch.utils.data import DataLoader, WeightedRandomSampler
from pytorch_metric_learning import samplers

from data_processing.productset import ProductSet
from config import config

SEED = config["seed"]
BATCH_SIZE = config["batch_size"]
USE_TFIDF = config["use_tfidf"]


def remove_non_ascii_chars(string: str) -> str:
    return re.sub(r"\\[a-z0-9]{3}", "", string)


def read_data(data_path, embedder, is_train=True):
    df = pd.read_csv(data_path)

    df["title_normalized"] = df["title"].apply(remove_non_ascii_chars) \
        .str.lower() \
        .str.replace(r"[^a-z0-9 ]", '', regex=True) \
        .str.replace(r" +", ' ', regex=True)

    df["embs"] = df["title_normalized"].apply(lambda x: embedder.get_sentence_vector(x))

    if USE_TFIDF:
        vectorizer = TfidfVectorizer(max_features=300)
        tfidf = vectorizer.fit_transform(df["title_normalized"])
        df["tfidf"] = pd.Series(tfidf.todense().tolist())

        vectorizer = CountVectorizer(max_features=300)
        tfidf = vectorizer.fit_transform(df["title_normalized"])
        df["count"] = pd.Series(tfidf.todense().tolist())

        df["embs"] = [row.tolist() for row in np.hstack([np.vstack(df["embs"].values), np.vstack(df["tfidf"].values), np.vstack(df["count"].values)])]

    if is_train:
        label_groups = df.groupby(["label_group"])["posting_id"].unique()
        df["matches"] = df["label_group"].map(label_groups.to_dict())
        df["matches"] = df["matches"].apply(lambda x: ' '.join(x))

        return df, label_groups.values

    return df


def get_dataloaders(data_path, embedder, mode="single", is_train=True):
    df, label_groups = read_data(data_path, embedder, is_train)

    train_ids, test_ids = train_test_split(np.concatenate(label_groups), test_size=0.2, shuffle=True, random_state=SEED)

    train_df = df[df["posting_id"].isin(train_ids)]
    test_df = df[df["posting_id"].isin(test_ids)]

    train_dataset = ProductSet(train_df, mode)
    test_dataset = ProductSet(test_df, mode)

    train_sampler = None
    test_sampler = None
    # if mode == "single":
    #     train_sampler = samplers.MPerClassSampler(train_df["label_group"].values, m=2, batch_size=BATCH_SIZE)
    #     test_sampler = samplers.MPerClassSampler(test_df["label_group"].values, m=2, batch_size=BATCH_SIZE)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_dataloader, test_dataloader
