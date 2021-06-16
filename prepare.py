import pandas as pd
import re
from mlxtend.preprocessing import TransactionEncoder


def sum_filter(ds, minSum, maxSum):
    if maxSum > 0:
        ds = ds.loc[(minSum <= ds['totalSum']) & (ds['totalSum'] <= maxSum)]
    return ds


def replace_char(ds):
    for i in ds.index:
        ds.at[i, 'items.name'] = ds.at[i, 'items.name'].replace('ё', 'е').replace('Ё', 'Е')
        ds.at[i, 'items.name'] = ds.at[i, 'items.name'].replace('й', 'и').replace('Й', 'И')
    return ds


def remove_space(ds):
    for i in ds.index:
        ds.at[i, 'items.name'] = ' '.join(ds.at[i, 'items.name'].split())
    return ds


def remove_phone(ds):
    regex_tel_number = r' \+?[7,8]?\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}\b'
    for i in ds.index:
        ds.at[i, 'items.name'] = re.sub(regex_tel_number, '', ds.at[i, 'items.name'])
    return ds


def remove_date(ds):
    regex_date = r'\d{2}.\d{2}.\d{4} \d{2}:\d{2}'
    for i in ds.index:
        ds.at[i, 'items.name'] = re.sub(regex_date, '', ds.at[i, 'items.name'])
    return ds


def remove_doc_num(ds):
    regex_dog_num = r' №\d*\b'
    for i in ds.index:
        ds.at[i, 'items.name'] = re.sub(regex_dog_num, '', ds.at[i, 'items.name'])
    return ds


def group(ds):
    grouped_prods = ds.groupby('fiscalDocumentNumber')['items.name'].apply(
        lambda group_series: group_series.tolist()).reset_index()
    groups_lists = grouped_prods['items.name'].values.tolist()
    data = list(filter(lambda x: len(x) > 2, groups_lists))

    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)
    data_frame = pd.DataFrame(te_ary, columns=te.columns_)

    return data_frame


def prepare(file, minSum, maxSum):
    dataset = pd.read_excel(file)
    ds = sum_filter(dataset, minSum, maxSum)
    ds = replace_char(ds)
    ds = remove_space(ds)
    ds = remove_phone(ds)
    ds = remove_date(ds)
    ds = remove_doc_num(ds)
    data_frame = group(ds)

    return data_frame