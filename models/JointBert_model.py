import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from .classifiers import *


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, intent_labels, slot_labels):
        super(JointBERT, self).__init__(config)
        self.num_intent_labels = len(intent_labels)
        self.num_slot_labels = len(slot_labels)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, dropout_rate=0.1)
        self.slot_classifier = SlotsClassifier(config.hidden_size, self.num_slot_labels, dropout_rate=0.1)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        # gettin' the pre-trained model outputs
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids)  # remember: token type ids refer to the sentence sgmentation
        # sequence_output, pooled_output, (hidden_states), (attentions) output structure
        sequence_out = bert_out.bert_out[0]
        pooled_out = bert_out[1]

        # logits of the classifications
        intent_logit = self.intent_classifier(pooled_out)
        slot_logit = self.slot_classifier(pooled_out)

        # computing the loss
        loss = 0

        # intent Softmax layer loss
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fun = nn.MSELoss()
                intent_loss = intent_loss_fun(intent_logit.view(-1), intent_label_ids.view(
                    -1))  # Tensor.view(-1) -> -1 auto infer the dimentions
            else:
                intent_loss_fun = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fun(intent_logit.view(-1, self.num_intent_labels), intent_label_ids.view(-1))

            loss += intent_loss

        # slots Softmax layer loss

        if slot_labels_ids is not None:
            # remember the possibility to implement crf
            slot_loss_fun = nn.CrossEntropyLoss(ignore_index=0)  # ignore index to skip the loss computation on
            # padding tokens, remeber that we have masked the padding tokens with 0
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logit.view(-1, self.num_intent_labels)[active_loss]
                active_labels = slot_labels_ids.view(-1)[active_loss]
                slot_loss = slot_loss_fun(active_logits, active_labels)
            else:
                slot_loss = slot_loss_fun(slot_logit.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))

            slot_loss_coef = 1.0  # if bad results has to be tuned, for now we set default fixed value
            loss += slot_loss * slot_loss

        output = ((intent_logit, slot_logit),) + bert_out[2:]

        output = (loss,) + output

        return output
