from os import listdir
from transformers import LineByLineTextDataset, PreTrainedTokenizer
import re
from typing import List


def read_book(path: str, block=1) -> List[str]:
    with open(path, encoding="utf8") as book:
        lines = book.readlines()
        lines = [" ".join(lines)]
        for sep in ".?!":
            new_lines = []
            for line in lines:
                line = re.sub('[^\w\s.!?,:\-()]', '', line)
                line = re.sub('\s+', ' ', line)
                split_line = line.split(sep)
                if len(split_line) > 1:
                    split_line = [l + sep for l in split_line[:-1]] + [split_line[-1]]
                    split_line = [l.strip() for l in split_line]
                new_lines += split_line
            lines = new_lines
        lines = [" ".join(el) for el in split_list(lines, block)]
        lines = [s for s in lines if len(s) > 20]
    return lines


def split_list(lst: List, n: int) -> List:
    new_list = []
    for i in range(0, len(lst)+1):
        if i % n == 0:
            new_list += [lst[i-n:i]]
    return new_list[1:]


def write_data(file_name, data):
    with open(file_name, 'w', encoding="utf8") as f:
        for i, line in enumerate(data):
            f.write(line + '\n')


def setup(block: int):
    text_data = [line for book in listdir('corpus') for line in read_book('corpus/' + book, block)]
    train_size = int(len(text_data) * 0.9)
    write_data('train_data.txt', text_data[:train_size])
    write_data('test_data.txt', text_data[train_size:])


def get_test_lines():
    with open('test_data.txt', encoding="utf8") as f:
        return f.readlines()


def get_dataset(data_file: str, tokenizer: PreTrainedTokenizer, block_size=128):
    return LineByLineTextDataset(tokenizer=tokenizer, file_path=data_file, block_size=block_size)


def get_train_dataset(tokenizer: PreTrainedTokenizer):
    return get_dataset('train_data.txt', tokenizer)


def get_eval_dataset(tokenizer: PreTrainedTokenizer):
    return get_dataset('test_data.txt', tokenizer)


if __name__ == "__main__":
    path = "corpus/dostoevskiy_tom_1.txt"
    lines = read_book(path, 2)
