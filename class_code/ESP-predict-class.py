import os
from sklearn import metrics
from glob import glob
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.bert_for_finetune_cpu import BertFinetuneCellCPU
from src.bert_for_finetune import BertFinetuneCell, BertCLS,BertCLSEval
from src.dataset import create_classification_dataset
from src.utils import LossCallBack,  BertLearningRate
from src.model_utils.config import config as args_opt, optimizer_cfg, bert_net_cfg
import numpy as np
import pandas as pd
from src import tokenization
_cur_dir = os.getcwd()

def sensitivity(Y_test,Y_pred,n=2):#n为分类数
    sen = []
    con_mat = metrics.confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)
    return sen

def specificity(Y_test,Y_pred,n=2):
    spe = []
    con_mat = metrics.confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    return spe

def truncate_seq_pair_1x(tokens_a, tokens_b, max_length):
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
        return tokens_a, tokens_b
    else:
        tokens_a=tokens_a[:max_length]
        tokens_b=tokens_b[:max_length-len(tokens_a)]
        return tokens_a,tokens_b

def truncate_seq_pair_2x(tokens_a, tokens_b, max_length):
    while len(tokens_a) + len(tokens_b) > max_length:
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    return tokens_a, tokens_b


def generate_predict_seq(predict_format,data,tokenizer,seq_len):
    if predict_format == "2x":
        tokens_a = list(data["seq"])
        tokens_b = list(data["smile"])
        id = str(data["id"])
    elif predict_format == "1x":
        tokens_a = list(data["seq"])
        tokens_b = list(data["smile"])
        id = str(data["id"])
    tokens_a = tokenizer.tokenize(tokens_a,is_smile=False)
    tokens_b = tokenizer.tokenize(tokens_b,is_smile=True)
    if predict_format == "1x":
        tokens_a, tokens_b = truncate_seq_pair_1x(tokens_a, tokens_b, seq_len - 3)
    elif predict_format == "2x":
        tokens_a, tokens_b = truncate_seq_pair_2x(tokens_a, tokens_b, seq_len - 3)
    assert len(tokens_a) + len(tokens_b) <= seq_len - 3

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if len(tokens_b)>0:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
    assert len(tokens) == len(segment_ids)
    input_ids = tokenization.convert_tokens_to_ids(args_opt.vocab_file, tokens)
    print(input_ids)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == seq_len
    assert len(input_mask) == seq_len
    assert len(segment_ids) == seq_len

    if "label" in data.keys():
        label_id = data["label"]
    else:
        label_id = -1
    return ms.Tensor([input_ids]),ms.Tensor([input_mask]),ms.Tensor([segment_ids]),ms.Tensor([[label_id]]),id,"".join(tokens_a),"".join(tokens_b)

def do_predict(seq_len=2048, network=None, num_class=2, load_checkpoint_path="",tokenizer=None):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_pretraining = network(bert_net_cfg, False, num_class)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)

    ds_predict = pd.read_csv(args_opt.data_url).to_dict("record")  # predict csv must have["id","seq"] or ["id_0","seq_0","id_1","seq_1"]
    data_file_name=args_opt.data_url.split("/")[-1].strip(".csv")

    predict_format="2x"

    softmax = ms.nn.Softmax()

    if args_opt.return_sequence==True or args_opt.return_csv==True:
        write_data=[]

    if args_opt.print_predict==True:
        true_labels=[]
        pred_labels=[]

    for data in ds_predict:
        input_ids, input_mask, token_type_id, label_ids,id,truncate_token_a,truncate_token_b=generate_predict_seq(predict_format,data,tokenizer,seq_len)
        logits,_,sequence_output,_,all_polled_output = model.predict(input_ids, input_mask, token_type_id, label_ids)
        logits=softmax(logits[0])
        data["pred_label"]=np.argmax(logits.asnumpy())

        if args_opt.print_predict == True:
            true_labels.append(data["label"])
            pred_labels.append(logits.asnumpy())

        if args_opt.return_sequence==True:
            data["truncate_0"]=truncate_token_a
            data["truncate_1"]=truncate_token_b
            data["feature"]=sequence_output.asnumpy()
        else:
            data["feature"]=None

        if args_opt.return_sequence == True or args_opt.return_csv == True:
            data["dense"] = logits.asnumpy()[1]
            print("ID: " + id + " Predict: " + str(data["pred_label"]))
            write_data.append(data)

            if args_opt.print_predict == True:
                true_labels.append(data["label"])
                pred_labels.append(logits)

            if args_opt.return_sequence == True or args_opt.return_csv == True:
                write_data.append(data)

    if args_opt.print_predict==True:
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

        print_result = {"model": load_checkpoint_path.split("/")[-1].strip(".ckpt"),"data":data_file_name}
        try:
            print_result["AUC"] = metrics.roc_auc_score(np.eye(num_class)[true_labels], pred_labels)
        except:
            print("[Warning] Can't caculate AUC, this may be due to the fact that the predicted category is only one category, CONTINUE")
        pred_labels = np.argmax(pred_labels, axis=1)
        print_result["ACC"] = metrics.accuracy_score(true_labels, pred_labels)
        print_result["precision"] = metrics.precision_score(true_labels, pred_labels)
        print_result["Recall"] = metrics.recall_score(true_labels, pred_labels)
        print_result["F1"] = metrics.f1_score(true_labels, pred_labels)
        print_result["MCC"] = metrics.matthews_corrcoef(true_labels, pred_labels)

        if num_class == 2:
            print_result["Sensitivity"] = sensitivity(true_labels, pred_labels, 2)[1]
            print_result["Specificity"] = specificity(true_labels, pred_labels, 2)[1]

        print("\n========================================")
        print(pd.DataFrame(print_result, index=["model"]))
        print("========================================\n")


    if args_opt.return_sequence==True:
        np.save(os.path.join(args_opt.output_url,data_file_name+"_predict_result.npy"),np.array(write_data))
    if args_opt.return_csv==True:
        pd.DataFrame(write_data).drop(["feature"],axis=1).to_csv(os.path.join(args_opt.output_url,data_file_name+"_predict_result_class.csv"))


def run_classifier():
    """run classifier task"""
    epoch_num = args_opt.epoch_num
    target = args_opt.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        context.set_context(enable_graph_kernel=True)
        if bert_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            bert_net_cfg.compute_type = mstype.float32
    elif target == "CPU":
        if args_opt.use_pynative_mode:
            context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", device_id=args_opt.device_id)
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target="CPU", device_id=args_opt.device_id)
    else:
        raise Exception("Target error, CPU or GPU or Ascend is supported.")


    netwithloss = BertCLS(bert_net_cfg, True, num_labels=args_opt.num_class, dropout_prob=0.1)

    if args_opt.do_predict==True:
        tokenizer = tokenization.FullTokenizer(vocab_file=args_opt.vocab_file,do_lower_case=False)
        finetune_ckpt_url = args_opt.load_checkpoint_url
        if args_opt.do_eval==False:
            if finetune_ckpt_url.endswith(".ckpt") ==False:
                raise "For predict, if do_eval==False, you should select only one checkpoint file and this file should end with .ckpt"
            else:
                best_ckpt=finetune_ckpt_url
        do_predict(bert_net_cfg.seq_length, BertCLSEval, args_opt.num_class, load_checkpoint_path=best_ckpt,
                   tokenizer=tokenizer)
    print("FINISH !!!")

if __name__ == "__main__":
    run_classifier()
