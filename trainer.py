from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    DataCollatorForLanguageModeling,
)
from training_arguments import TrainingArguments
import dataset


def train(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, train_batch_size: int, eval_batch_size: int, mlm=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=mlm, mlm_probability=0.15
    )
    args = TrainingArguments()
    args.per_gpu_train_batch_size = train_batch_size
    args.per_gpu_eval_batch_size = eval_batch_size

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=dataset.get_train_dataset(tokenizer),
        eval_dataset=dataset.get_eval_dataset(tokenizer),
        prediction_loss_only=True,
    )

    trainer.train(model_path=None)
    trainer.save_model()
