import csv
import json


class DataProcessor():
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, label_vocab):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f_obj:
            reader = csv.reader(f_obj, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _read_text(self, input_file):
        lines = []
        with open(input_file, 'r') as f_obj:
            words = []
            labels = []
            for line in f_obj:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    def _read_json(self, input_file):
        lines = []
        with open(input_file, 'r') as f_obj:
            for line in f_obj:
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label', None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index +
                                                     1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'sent-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index +
                                           1] = ['I-' + key
                                                 ] * (len(sub_name) - 1)
                lines.append({"words": words, "labels": labels})
        return lines
