"""Generate MindRecord for BERT finetuning runner: TNEWS."""
import os
import pandas as pd
import random
random.seed(100)
import json
import csv
from argparse import ArgumentParser
import six
import numpy as np

from mindspore.mindrecord import FileWriter
import mindspore.dataset as ds

import sys
sys.path.append("..")
import tokenization


def parse_args():
    parser = ArgumentParser(description="Generate MindRecord for bert task: Protein")
    parser.add_argument("--data_dir", type=str, default="./")
    parser.add_argument("--task_name", type=str, default="tnews", help="The name of the task to train.")
    parser.add_argument("--vocab_file", type=str, default="./vocab_smile.txt",
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="The output directory where the mindrecord will be written.")
    parser.add_argument("--do_lower_case", type=bool, default=True, help="Whether to lower case the input text. "
                                                                     "Should be True for uncased models "
                                                                     "and False for cased models.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--do_train", type=bool, default=True, help="Whether to run training.")
    parser.add_argument("--do_val", type=bool, default=True, help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", type=bool, default=True, help="Whether to run eval on the dev set.")

    args_opt = parser.parse_args()
    return args_opt


class PaddingInputExample():
    """Fake example so the num input examples is a multiple of the batch size.
    This is used for padding purposes in TPU/GPU distributed settings.
    """


class InputFeatures():
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {str(label): i for i, label in enumerate(label_list)}

    # 将 seq 作为 text_a，smile 作为 text_b
    tokens_a = tokenizer.tokenize(example.text_a)  # 序列按字符切
    tokens_b = tokenizer.tokenize(example.text_b, is_smile=True)

    if tokens_b:
        tokens_a, tokens_b = truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]
    print(tokens_a)
    print(tokens_b)

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenization.convert_tokens_to_ids(args.vocab_file, tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    return InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a MindRecord file."""
    schema = {
        "input_ids": {"type": "int32", "shape": [-1]},
        "input_mask": {"type": "int32", "shape": [-1]},
        "segment_ids": {"type": "int32", "shape": [-1]},
        "label_ids": {"type": "int32", "shape": [-1]},
        "is_real_example": {"type": "int32", "shape": [-1]},
    }
    writer = FileWriter(output_file, overwrite=True)
    writer.add_schema(schema)
    total_written = 0
    random.shuffle(examples)
    for ex_index, example in enumerate(examples):
        all_data = []
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

        input_ids = np.array(feature.input_ids, dtype=np.int32)
        input_mask = np.array(feature.input_mask, dtype=np.int32)
        segment_ids = np.array(feature.segment_ids, dtype=np.int32)
        label_ids = np.array(feature.label_id, dtype=np.int32)
        is_real_example = np.array(feature.is_real_example, dtype=np.int32)
        data = {'input_ids': input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "label_ids": label_ids,
                "is_real_example": is_real_example}
        all_data.append(data)
        if all_data:
            writer.write_raw_data(all_data)
            total_written += 1
    writer.commit()
    print("Total instances is: ", total_written, flush=True)


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
        return tokens_a, tokens_b
    else:
        tokens_a = tokens_a[:max_length]
        tokens_b = tokens_b[:max_length-len(tokens_a)]
        return tokens_a, tokens_b


class DataProcessor():
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_val_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file):
        lines = pd.read_csv(input_file)
        return lines


class InputExample():
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class TnewsProcessor(DataProcessor):
    """Processor for the dataset."""
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_val_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "val")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def _create_examples(self, lines, set_type):
        examples = []
        for i in range(len(lines)):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(lines['seq'][i])  # 将 seq 作为句子 A
            text_b = tokenization.convert_to_unicode(lines['smile'][i])  # 将 smile 作为句子 B
            label = tokenization.convert_to_unicode(str(lines['label'][i]))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def main():
    processors = {
        "tnews": TnewsProcessor,
    }

    if not args.do_train and not args.do_val and not args.do_test:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_test' must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = list(set(list(pd.read_csv(os.path.join(args.data_dir, "train.csv"))["label"])))
    label_list.sort()

    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        train_file = os.path.join(args.output_dir, "train.mindrecord")
        file_based_convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer, train_file)

    if args.do_val:
        val_examples = processor.get_val_examples(args.data_dir)
        val_file = os.path.join(args.output_dir, "val.mindrecord")
        file_based_convert_examples_to_features(val_examples, label_list, args.max_seq_length, tokenizer, val_file)

    if args.do_test:
        predict_examples = processor.get_test_examples(args.data_dir)
        predict_file = os.path.join(args.output_dir, "test.mindrecord")
        file_based_convert_examples_to_features(predict_examples, label_list, args.max_seq_length, tokenizer, predict_file)


if __name__ == "__main__":
        args = parse_args()
        main()
