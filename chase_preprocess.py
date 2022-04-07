import json
import os
import glob
import shutil
from tqdm import tqdm
import re


def split_query(query):
    query_new = ' '.join(re.split(r"([,()><=])", query))
    return query_new.split()


def main():
    chase_path = './data/chase'

    names = ['train', 'dev']

    for file_name in names:

        with open(os.path.join(chase_path, 'chase_' + file_name + '.json')) as f:
            chase_data = json.load(f)

        datalist = []

        for ex in tqdm(chase_data, desc='preprocess {}'.format(file_name)):
            for ex_i in ex['interaction']:
                data_unit = {'db_id': ex['database_id'], 'query': ex_i['query'],
                             'query_toks': split_query(ex_i['query']),
                             'query_toks_no_value': ex_i['query_toks_no_value'], 'sql': ex_i['sql'],
                             'question': ex_i['utterance'], 'question_toks': ex_i['utterance_toks']}
                datalist.append(data_unit)

        with open(os.path.join(chase_path, file_name + '.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(datalist, indent=4, separators=(',', ': '), ensure_ascii=False))


def db_mv():
    db_path = './data/database'
    db = glob.glob(os.path.join(db_path, '*.sqlite'))
    for i in tqdm(range(len(db)), desc='Processing'):
        sql_path = db[i]
        db_name = sql_path.split('/')[-1].split('.')[0]
        os.mkdir(os.path.join(db_path, db_name))
        shutil.move(sql_path, os.path.join(db_path, db_name))


if __name__ == '__main__':
    main()

