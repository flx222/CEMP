'''
Bert for finetune script.
'''


from .utils import CrossEntropyCalculation,FocalLossCalculation
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore.ops import Concat as C
from .bert_model import BertModel
import mindspore.numpy as mnp

# GRADIENT_CLIP_TYPE = 1
# GRADIENT_CLIP_VALUE = 1.0
# grad_scale = C.MultitypeFuncGraph("grad_scale")
# reciprocal = P.Reciprocal()
#
#
# @grad_scale.register("Tensor", "Tensor")
# def tensor_grad_scale(scale, grad):
#     return grad * reciprocal(scale)
#
#
# _grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
# grad_overflow = P.FloatStatus()
#
#
# @_grad_overflow.register("Tensor")
# def _tensor_grad_overflow(grad):
#     return grad_overflow(grad)



class BertCLS(nn.Cell):
    """
    Train interface for classification finetuning task.
    """

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertCLS, self).__init__()
        self.bert = BertCLSModel(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings,
                                 assessment_method)
        self.loss = CrossEntropyCalculation(is_training)
        self.num_labels = num_labels
        self.assessment_method = assessment_method
        self.is_training = is_training

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits = self.bert(input_ids, input_mask, token_type_id)
        if self.assessment_method == "spearman_correlation":
            if self.is_training:
                loss = self.loss(logits, label_ids)
            else:
                loss = logits
        else:
            loss = self.loss(logits, label_ids, self.num_labels)
        return loss



class BertCLSModel(nn.Cell):
    """
    This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
    LCQMC(num_labels=2), Chnsenti(num_labels=2). The returned output represents the final
    logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertCLSModel, self).__init__()
        self.log_softmax = P.LogSoftmax(axis=-1)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
            self.log_softmax= P.Softmax(axis=-1)
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)

        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1.0 - dropout_prob)
        self.assessment_method = assessment_method

        self.lstm_hidden_size = config.hidden_size // 2
        self.lstm = nn.LSTM(config.hidden_size, self.lstm_hidden_size,num_layers=2, batch_first=True, bidirectional=True)

    def construct(self, input_ids, input_mask, token_type_id):
        sequence_output, pooled_output, _ = self.bert(input_ids, token_type_id, input_mask)
        sequence_output = self.cast(sequence_output, self.dtype)
        sequence_output = self.dropout(sequence_output)
        batch_size = input_ids.shape[0]
        data_type = self.dtype
        hidden_size = self.lstm_hidden_size
        # h0 = P.Zeros()((2, batch_size, hidden_size), data_type)
        # c0 = P.Zeros()((2, batch_size, hidden_size), data_type)
        _, (hidden, _) = self.lstm(sequence_output)
        hidden = self.dropout(hidden)
        hidden = self.dropout(mnp.concatenate((hidden[-2, :, :], hidden[-1, :, :]), axis=1))
        logits = self.dense_1(hidden)
        logits = self.cast(logits, self.dtype)
        logits = self.log_softmax(logits)
        return logits