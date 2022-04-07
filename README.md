## Introduction

This repository contains the code for the whole process in training a model which can predict natural language question from 'SQL'+'schema'.

The task mainly concerns on problems related to Chinese Text-to-SQL.

## Environment Prepare

`pip install -r requirement.txt`

## Dataset Prepare

First we need to get the origin training data from [Chase](https://github.com/xjtu-intsoft/chase/tree/page/data).

As we use preprocess kit from [Spider](https://github.com/taoyds/spider), so we have to preprocess the Chase dataset format into the format of Spider  dataset.

When we unzip `Chase.zip` and `database.zip` into our work space, we need to put them into `data` directory, as the file tree given.

```shell
sql2nl
├── README.md
├── data
    └── chase
        ├── database
        ├── dev.json
        ├── tables.json
        ├── test.json
        └── train.json
```

You should rename the origin chase dataset file like this:
```shell
chase_table.json -> tables.json
chase_train.json -> train.json
chase_dev.json -> dev.json
chase_test.json -> test.json
```

And then when you get `database` directory, you need to run file rebase script so as to make the file struct looks like `Spider-database`.

`python database_move.py`

Finally we need to preprocess the train and dev dataset.

`python chase_preprocess.py`

## Model Prepare

In this code we use [mT5](https://github.com/google-research/multilingual-t5) as our pretrain model.  You can download the mT5-small version from [huggingface](https://huggingface.co/google/mt5-small/tree/main).

Put the mt5-small directory into cache directory. The dict structure should look like this `./cache/mt5-small`


## Preprocess
Step 1: Preprocess via adding schema-linking and value-linking tag.

`python pre1_schema_linking.py`

Step 2: Building the input(`.src`) and output(`.tgt`) for mT5.

`python pre2_serialization.py`

## Train

Just run the training script and you will get checkpoint file.

`python train.py`

In this training step we use [transformers library](https://github.com/huggingface/transformers) provided by huggingface.