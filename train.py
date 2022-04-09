from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import numpy as np
import argparse
import glob

import os


def prepare_translation_datasets(data_paths):
    train_data = []
    test_data = []

    for data_path in data_paths:
        with open(os.path.join(data_path, "train.src"), "r", encoding="utf-8") as f:
            sql_text = f.readlines()
            sql_text = [text.strip("\n") for text in sql_text]

        with open(os.path.join(data_path, "train.tgt"), "r") as f:
            nl_text = f.readlines()
            nl_text = [text.strip("\n") for text in nl_text]

        for sql, nl in zip(sql_text, nl_text):
            train_data.append({'sql': sql, 'nl': nl})

        with open(os.path.join(data_path, "dev.src"), "r", encoding="utf-8") as f:
            sql_text = f.readlines()
            sql_text = [text.strip("\n") for text in sql_text]

        with open(os.path.join(data_path, "dev.tgt"), "r") as f:
            nl_text = f.readlines()
            nl_text = [text.strip("\n") for text in nl_text]

        for sql, nl in zip(sql_text, nl_text):
            test_data.append({'sql': sql, 'nl': nl})

    train_dict = {'id': np.arange(len(train_data)),
                  'translation': train_data}
    # create dataset
    train_ds = Dataset.from_dict(mapping=train_dict)

    test_dict = {'id': np.arange(len(test_data)),
                 'translation': test_data}
    # create dataset
    test_ds = Dataset.from_dict(mapping=test_dict)
    dataset = DatasetDict()
    dataset['train'] = train_ds
    dataset['test'] = test_ds

    return dataset


def train(path):
    nlsql = prepare_translation_datasets(path)

    tokenizer = AutoTokenizer.from_pretrained("cache/mt5-small")

    source_lang = "sql"
    target_lang = "nl"
    prefix = "translate sql to nl: "

    def preprocess_function(examples):
        inputs = [prefix + example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_nlsql = nlsql.map(preprocess_function, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained("cache/mt5-small")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_nlsql["train"],
        eval_dataset=tokenized_nlsql["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    # trainer.save_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default='./dataset_post')
    args = parser.parse_args()
    path = args.dataset_path

    paths = glob.glob(os.path.join(path, '*'))
    train(paths)
