from torch import nn
import torch.nn.functional as F
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

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None):
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


# reference: <<Exploring Pre-trained Language Models for Event Extraction and Generation>>
class BertSpan(BertPreTrainedModel):
    # TODO: add decision boundary for model.
    def __init__(self, config, num_role):
        super(BertSpan, self).__init__()
        self.num_role = num_role
        self.model = BertModel(config)
        # Full-connect layer for start.
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_classifier = self.linear(config.hidden_size, num_role)
        self.end_classifier = self.linear(config.hidden_size, num_role)
        # TODO: re-weight loss for roles in section 3.4
        self.criterion = nn.BCEWithLogitsLoss()
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        start_labels=None,
        end_labels=None,
    ):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = outputs[0]
        # sequence_output: (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)

        # start_preds: (batch_size, seq_len, num_role)
        # Note that
        start_logits = self.start_classifier(sequence_output)
        end_logits = self.end_classifier(sequence_output)
        role_span_lists = self._arg_span_determine(
            start_logits,
            end_logits,
            attention_mask,
        )
        outputs = (role_span_lists, )
        if start_labels is not None and end_labels is not None:
            loss_start = self.criterion(start_logits, start_labels)
            loss_end = self.criterion(end_logits, end_labels)
            outputs = (loss_start + loss_end, ) + outputs
        return outputs

    def _arg_span_determine(
        self,
        start_logits,
        end_logits,
        attention_masks,
        boundary=0.5,
    ):
        "Reference: section 3.3 Argument Span Determination"
        # The idx of '[SEP]' in each batch.
        # role_span_list: (batch_size, num_role, num_span)
        batch_size, _ = attention_masks.shape
        role_span_lists = [[[] for _ in range(self.num_role)]
                           for _ in range(batch_size)]
        last_valid_idx = attention_masks.int().sum(1) - 1

        # (batch_size, seq_len, num_roles)
        p_role_start = start_logits.sigmoid()
        p_role_end = end_logits.sigmoid()

        # TODO: refactor in matrix form
        for b_idx in range(batch_size):
            for role in range(self.num_role):
                state, a_s, a_e = 1, -1, -1
                prob_start = p_role_start[b_idx][role]
                prob_end = p_role_end[b_idx][role]
                # Code below is an automaton.
                for seq_idx in range(last_valid_idx[b_idx]):
                    if  prob_start[seq_idx] > boundary \
                            and state == 1:
                        a_s, state = seq_idx, 2

                    if state == 2:
                        if prob_start[seq_idx] > prob_start[a_s] \
                                and seq_idx != a_s:
                            a_s = seq_idx

                        if prob_end[seq_idx] > boundary:
                            a_e, state = seq_idx, 3

                    if state == 3:
                        if prob_end[seq_idx] > prob_end[a_e] and seq_idx != a_e:
                            a_e = seq_idx

                        if prob_start[seq_idx] > boundary and seq_idx != a_s:
                            role_span_lists[b_idx][role].append([a_s, a_e])
                            state, a_s, a_e = 2, seq_idx, -1

        return role_span_lists
