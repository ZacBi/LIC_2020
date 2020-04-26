from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities


class SeqEntityScore():
    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.y_preds = []
        self.y_trues = []

    def get_result(self):
        # pylint: disable=invalid-name
        f1 = f1_score(self.y_trues, self.y_preds)
        precision = precision_score(self.y_trues, self.y_preds)
        recall = recall_score(self.y_trues, self.y_preds)

        return {'f1': f1, 'precision': precision, 'recall': recall}

    def update(self, y_true, y_pred):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        y_true = [self.id2label[label_id] for label_id in y_true]
        y_pred = [self.id2label[label_id] for label_id in y_pred]
        self.y_preds.append(y_pred)
        self.y_trues.append(y_true)

    @classmethod
    def get_entities(cls, id2label, label_id_seq):
        seq = [id2label[label_id] for label_id in label_id_seq]
        return get_entities(seq)
