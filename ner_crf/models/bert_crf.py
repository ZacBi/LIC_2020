# pylint: disable=bad-continuation

from torch import nn
from transformers import BertPreTrainedModel
from transformers import BertModel
from ner_crf.models.crf import CRF


class BertCRF(BertPreTrainedModel):
    def __init__(self, config, label2id, cls_token='[CLS]', sep_token='[SEP]'):
        super(BertCRF, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, len(label2id))
        self.crf = CRF(
            num_labels=len(label2id),
            start_tag_id=label2id[cls_token],
            end_tag_id=label2id[sep_token],
        )
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
    ):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        outputs = (emissions, )
        if labels is not None:
            loss = self.crf(emissions, tags=labels, mask=attention_mask)
            outputs = (loss, ) + outputs
        return outputs  # (loss), scores

    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask)
