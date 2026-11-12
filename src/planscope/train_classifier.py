'''
Fine-tune DistilBERT to predict an application's category from its description.
Keeps the epoch with the best validation macro-F1 and saves it to models/category-distilbert/.
'''

from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from planscope.evaluate import macro_f1

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data' / 'processed' / 'category'
OUT_DIR = ROOT / 'models' / 'category-distilbert'
CHECKPOINT_DIR = ROOT / 'models' / 'checkpoints'
BASE_MODEL = 'distilbert-base-uncased'
MAX_TOKENS = 128  # 95% of descriptions fit; the category signal is at the start


def to_dataset(df, tokenizer, label2id):
    '''Tokenise the texts and attach integer labels, as a Hugging Face Dataset.'''
    data = Dataset.from_pandas(pd.DataFrame({'text': df['text'], 'label': df['category'].map(label2id)}))
    return data.map(lambda batch: tokenizer(batch['text'], truncation=True, max_length=MAX_TOKENS), batched=True, remove_columns=['text'])


def compute_metrics(prediction):
    '''Validation macro-F1 from the model's raw scores (the highest score is the predicted category).'''
    return {'macro_f1': macro_f1(prediction.label_ids, prediction.predictions.argmax(axis=1))}


def main():
    if not torch.cuda.is_available():
        raise SystemExit('No CUDA GPU found. Install the CUDA build of PyTorch first.')

    train, val = (pd.read_parquet(DATA_DIR / f'{name}.parquet') for name in ['train', 'val'])
    labels = sorted(train['category'].unique())
    label2id = {label: i for i, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=2,
        learning_rate=5e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=256,
        train_sampling_strategy='group_by_length',
        bf16=True,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        logging_steps=500,
        report_to='none',
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=to_dataset(train, tokenizer, label2id),
        eval_dataset=to_dataset(val, tokenizer, label2id),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    print(trainer.evaluate())


if __name__ == '__main__':
    main()