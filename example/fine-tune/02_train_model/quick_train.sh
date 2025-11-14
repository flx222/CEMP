run_task () {
    local fold=$1
    local data_url="./example/fine-tune/dataset/fold_${fold}"
    local checkpoint_url="./pretrain_model/pretrained_smile.ckpt"
    local output_url="${data_url}"

    # Train
    python mpbert_regress_smile.py \
        --config_path config_2048_sm.yaml \
        --data_url "${data_url}" \
        --load_checkpoint_url "${checkpoint_url}" \
        --output_url "${output_url}" \
        --device_id 5 \
        --do_train True \
        --description regress \
        1> "${data_url}/train.log" 2> "${data_url}/train_sys.log"

    # Evaluate
    python mpbert_regress_smile.py \
        --config_path config_2048_sm.yaml \
        --data_url "${data_url}" \
        --load_checkpoint_url "${data_url}" \
        --output_url "${output_url}" \
        --device_id 5 \
        --do_eval True \
        --description regress \
        1> "${data_url}/eval.log" 2> "${data_url}/eval_sys.log"
}
for i in {0..0}; do
    run_task $i
done
