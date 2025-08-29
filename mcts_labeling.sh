#!/bin/bash

data_file=$2
total_parts=$4
llm=$6
max_expansion_num=$8
iterations=${10}
threshold=${12}
temperature=${14}
retriever_ip=${16}
retriever_port=${18}
local_llm_ip=${20}
local_llm_port=${22}

current_time=$(date "+%d_%H_%M")
mkdir log
# 对 split_index 进行循环，从 start_index 到 end_index
for ((split_index=0; split_index<=${total_parts}-1; split_index++))
do
    nohup python mcts_labeling.py \
    --data_file ${data_file}.jsonl \
    --split_index $split_index \
    --total_parts $total_parts \
    --llm $llm \
    --max_expansion_num $max_expansion_num \
    --iterations $iterations \
    --threshold $threshold \
    --temperature $temperature \
    --retriever_port $retriever_port \
    --retriever_ip $retriever_ip \
    --local_llm_ip  $local_llm_ip \
    --local_llm_port $local_llm_port \
    > log/${current_time}_${data_file}_${split_index}_${llm}_${max_expansion_num}_${iterations}_${threshold}.log 2>&1 &
done
