#!/bin/bash

DATASET=${2}
MODEL_PATH=${4}
BEAM_DEPTH=${6}
EXPAND_NUM_THOUGHT=${8}
BEAM_SIZE_THOUGHT=${10}
EXPAND_NUM_SEARCH=${12}
BEAM_SIZE_SEARCH=${14}
TEMPERATURE=${16}
POLICY_IP=${18}
POLICY_PORT=${20}
VALUE_IP=${22}
VALUE_PORT=${24}
RETRIEVER_IP=${26}
RETRIEVER_PORT=${28}

python inference.py \
--model_path save/$MODEL_PATH \
--dataset $DATASET \
--beam_size_thought $BEAM_SIZE_THOUGHT \
--beam_size_search $BEAM_SIZE_SEARCH \
--beam_depth $BEAM_DEPTH \
--expand_num_thought $EXPAND_NUM_THOUGHT \
--expand_num_search $EXPAND_NUM_SEARCH \
--temperature $TEMPERATURE \
--policy_ip $POLICY_IP \
--policy_port $POLICY_PORT \
--value_ip $VALUE_IP \
--value_port $VALUE_PORT \
--retriever_ip $RETRIEVER_IP \
--retriever_port $RETRIEVER_PORT