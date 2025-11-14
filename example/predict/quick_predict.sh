# 定义常用变量
config_path="config_2048_sm.yaml"
vocab_file="./vocab_smile.txt"
device_id=7
description="regress"

for i in {0..0}; do
    data_url="./example/fine-tune/dataset/fold_0/test.csv"
    checkpoint_url="./example/fine-tune/dataset/fold_0/Best_Model_Num_*.ckpt"
    output_url="./example/fine-tune/dataset/fold_0/"
    log_file="${output_url}/pred.log"
    sys_log_file="${output_url}/pred_sys.log"

    python mpbert_regress_smile.py \
        --config_path ${config_path} \
        --data_url ${data_url} \
        --load_checkpoint_url ${checkpoint_url} \
        --vocab_file ${vocab_file} \
        --output_url ${output_url} \
        --device_id ${device_id} \
        --do_predict True \
        --return_csv True \
        --description ${description} \
        1> ${log_file} 2> ${sys_log_file}
done
