from os import listdir
from transformers import PreTrainedTokenizer, TextDataset, AutoTokenizer
from typing import List
import torch


def read_book(path: str) -> List[str]:
    with open(path, encoding="utf8") as book:
        return book.readlines()


def setup():
    text_data = [line for book in listdir('corpus') for line in read_book('corpus/' + book)]
    text_data = [line.strip() for line in text_data]
    text_data = [line for line in text_data if len(line.split(" ")) > 1]
    with open("corpus.txt", "w", encoding="utf8") as f:
        f.writelines(text_data)


def get_dataset(tokenizer: PreTrainedTokenizer, block_size=128):
    full_dataset = TextDataset(tokenizer=tokenizer, file_path='corpus.txt', block_size=block_size)
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset


if __name__ == "__main__":
    setup()
    tokenizer = AutoTokenizer.from_pretrained("xlm-mlm-xnli15-1024")
    train_dataset, test_dataset = get_dataset(tokenizer)

