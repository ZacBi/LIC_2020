from torch import nn
from transformers import BertPreTrainedModel
from transformers import BertModel
from ner_crf.models.crf import CRF


class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config, label2id, device):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, len(label2id))
        self.crf = CRF(tagset_size=len(label2id),
                       tag_dictionary=label2id,
                       device=device,
                       is_bert=True)
        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                input_lens=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits, )
        if labels is not None:
            loss = self.crf.calculate_loss(logits,
                                           tag_list=labels,
                                           lengths=input_lens)
            outputs = (loss, ) + outputs
        return outputs  # (loss), scores
