
#######################################################################################################################################################################################################################################
## 7B

CUDA_VISIBLE_DEVICES=1 nohup /mnt/workspace/sunhao/miniconda3/envs/mcts/bin/python retriever_server.py --port 5002 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup /mnt/workspace/sunhao/miniconda3/envs/mcts/bin/python value_model_server.py --port 7000 --base_device 0 --model_path /mnt/workspace/sunhao/code/prm-trainning/save/02_05_02_35_mcts_v3_hierarchical_lr_5e-6_epoch_15 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 /mnt/workspace/sunhao/miniconda3/envs/mcts/bin/python -m sglang.launch_server --model-path /mnt/workspace/sunhao/code/prm-trainning/save/02_05_02_35_mcts_v3_hierarchical_lr_5e-6_epoch_15 --host 0.0.0.0 --tp 1 --dp 4 --port 6002 2>&1 &


##################################################################################################################################################################################################################################################
## 14B

CUDA_VISIBLE_DEVICES=0 nohup /mnt/workspace/sunhao/miniconda3/envs/myenv/bin/python retriever_server.py --port 5002 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup /mnt/workspace/sunhao/miniconda3/envs/mcts/bin/python value_model_server_hierarchical.py --port 7000 --base_device 0 --model_path /mnt/workspace/sunhao/code/prm-trainning/save/02_07_14_30_mcts_v3_hierarchical_lr_5e-6_epoch_15 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup /mnt/workspace/sunhao/miniconda3/envs/mcts/bin/python  value_model_server_hierarchical.py --port 7001 --base_device 0 --model_path /mnt/workspace/sunhao/code/prm-trainning/save/02_07_14_30_mcts_v3_hierarchical_lr_5e-6_epoch_15 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup /mnt/workspace/sunhao/miniconda3/envs/mcts/bin/python  value_model_server_hierarchical.py --port 7002 --base_device 0 --model_path /mnt/workspace/sunhao/code/prm-trainning/save/02_07_14_30_mcts_v3_hierarchical_lr_5e-6_epoch_15 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup /mnt/workspace/sunhao/miniconda3/envs/mcts/bin/python  value_model_server_hierarchical.py --port 7003 --base_device 0 --model_path /mnt/workspace/sunhao/code/prm-trainning/save/02_07_14_30_mcts_v3_hierarchical_lr_5e-6_epoch_15 2>&1 &


CUDA_VISIBLE_DEVICES=4,5,6,7 /mnt/workspace/sunhao/miniconda3/envs/mcts/bin/python -m sglang.launch_server --model-path /mnt/workspace/sunhao/code/prm-trainning/save/02_07_14_30_mcts_v3_hierarchical_lr_5e-6_epoch_15 --host 0.0.0.0 --tp 1 --dp 4 --port 6002 2>&1 &



#######################################################################################################################################################################################################################################
## baseline BEAM_SIZE_THOUGHT 100 EXPAND_NUM_SEARCH 1
## 7B
model_path=02_05_02_35_mcts_v3_hierarchical_lr_5e-6_epoch_15
ip=22.11.219.113


nohup bash parallel_inference_72b_prm.sh DATASET nq MODEL_PATH $model_path TOTAL_PARTS 8 DEVICES 0,1,2,3,4,5,6,7 BEAM_DEPTH 10 EXPAND_NUM_THOUGHT 3 BEAM_SIZE_THOUGHT 1 EXPAND_NUM_SEARCH 3 BEAM_SIZE_SEARCH 1 TEMPERATURE 0.7 LOG_PORT 6001 SEARCH_LOG_PORT 7001 POLICY_IP $ip POLICY_PORT 6002 ARM_IP none ARM_PORT 0 VALUE_IP $ip VALUE_PORT 7002 RETRIEVER_IP $ip RETRIEVER_PORT 5002 2>&1 &

nohup bash parallel_inference_72b_hierarchical.sh DATASET tqa MODEL_PATH $model_path TOTAL_PARTS 8 DEVICES 0,1,2,3,4,5,6,7 BEAM_DEPTH 10 EXPAND_NUM_THOUGHT 3 BEAM_SIZE_THOUGHT 1 EXPAND_NUM_SEARCH 3 BEAM_SIZE_SEARCH 1 TEMPERATURE 0.7 LOG_PORT 6002 SEARCH_LOG_PORT 7001 POLICY_IP $ip POLICY_PORT 6002 ARM_IP none ARM_PORT 0 VALUE_IP $ip VALUE_PORT 7003 RETRIEVER_IP 22.5.226.141 RETRIEVER_PORT 5003 2>&1 &

nohup bash parallel_inference_72b_hierarchical.sh DATASET 2wikimultihopqa MODEL_PATH $model_path TOTAL_PARTS 8 DEVICES 0,1,2,3,4,5,6,7 BEAM_DEPTH 10 EXPAND_NUM_THOUGHT 3 BEAM_SIZE_THOUGHT 1 EXPAND_NUM_SEARCH 3 BEAM_SIZE_SEARCH 1 TEMPERATURE 0.7 LOG_PORT 6003 SEARCH_LOG_PORT 7001 POLICY_IP $ip POLICY_PORT 6002 ARM_IP none ARM_PORT 0 VALUE_IP $ip VALUE_PORT 7000 RETRIEVER_IP 22.5.226.141 RETRIEVER_PORT 5004 2>&1 &

nohup bash parallel_inference_72b_hierarchical.sh DATASET bamboogle MODEL_PATH $model_path TOTAL_PARTS 8 DEVICES 0,1,2,3,4,5,6,7 BEAM_DEPTH 10 EXPAND_NUM_THOUGHT 3 BEAM_SIZE_THOUGHT 1 EXPAND_NUM_SEARCH 3 BEAM_SIZE_SEARCH 1 TEMPERATURE 0.7 LOG_PORT 6004 SEARCH_LOG_PORT 7001 POLICY_IP $ip POLICY_PORT 6002 ARM_IP none ARM_PORT 0 VALUE_IP $ip VALUE_PORT 7001 RETRIEVER_IP 22.5.226.141 RETRIEVER_PORT 5005 2>&1 &

nohup bash parallel_inference_72b_hierarchical.sh DATASET bamboogle MODEL_PATH $model_path TOTAL_PARTS 8 DEVICES 0,1,2,3,4,5,6,7 BEAM_DEPTH 10 EXPAND_NUM_THOUGHT 1 BEAM_SIZE_THOUGHT 1 EXPAND_NUM_SEARCH 1 BEAM_SIZE_SEARCH 1 TEMPERATURE 0.7 LOG_PORT 6005 SEARCH_LOG_PORT 7001 POLICY_IP $ip POLICY_PORT 6002 ARM_IP none ARM_PORT 0 VALUE_IP $ip VALUE_PORT 7001 RETRIEVER_IP 22.5.226.141 RETRIEVER_PORT 5005 2>&1 &

#######################################################################################################################################################################################################################################

## baseline BEAM_SIZE_THOUGHT 100 EXPAND_NUM_SEARCH 1
## 14B

model_path=02_07_14_30_mcts_v3_hierarchical_lr_5e-6_epoch_15
ip=22.5.239.225


nohup bash parallel_inference_72b_hierarchical.sh DATASET nq MODEL_PATH $model_path TOTAL_PARTS 8 DEVICES 0,1,2,3,4,5,6,7 BEAM_DEPTH 10 EXPAND_NUM_THOUGHT 3 BEAM_SIZE_THOUGHT 1 EXPAND_NUM_SEARCH 3 BEAM_SIZE_SEARCH 1 TEMPERATURE 0.7 LOG_PORT 6001 SEARCH_LOG_PORT 7001 POLICY_IP $ip POLICY_PORT 6002 ARM_IP none ARM_PORT 0 VALUE_IP $ip VALUE_PORT 7002 RETRIEVER_IP 22.5.226.141 RETRIEVER_PORT 5002 2>&1 &

nohup bash parallel_inference_72b_hierarchical.sh DATASET tqa MODEL_PATH $model_path TOTAL_PARTS 8 DEVICES 0,1,2,3,4,5,6,7 BEAM_DEPTH 10 EXPAND_NUM_THOUGHT 3 BEAM_SIZE_THOUGHT 1 EXPAND_NUM_SEARCH 3 BEAM_SIZE_SEARCH 1 TEMPERATURE 0.7 LOG_PORT 6002 SEARCH_LOG_PORT 7001 POLICY_IP $ip POLICY_PORT 6002 ARM_IP none ARM_PORT 0 VALUE_IP $ip VALUE_PORT 7003 RETRIEVER_IP 22.5.226.141 RETRIEVER_PORT 5003 2>&1 &

nohup bash parallel_inference_72b_hierarchical.sh DATASET 2wikimultihopqa MODEL_PATH $model_path TOTAL_PARTS 8 DEVICES 0,1,2,3,4,5,6,7 BEAM_DEPTH 10 EXPAND_NUM_THOUGHT 3 BEAM_SIZE_THOUGHT 1 EXPAND_NUM_SEARCH 3 BEAM_SIZE_SEARCH 1 TEMPERATURE 0.7 LOG_PORT 6003 SEARCH_LOG_PORT 7001 POLICY_IP $ip POLICY_PORT 6002 ARM_IP none ARM_PORT 0 VALUE_IP $ip VALUE_PORT 7000 RETRIEVER_IP 22.5.226.141 RETRIEVER_PORT 5004 2>&1 &

nohup bash parallel_inference_72b_hierarchical.sh DATASET bamboogle MODEL_PATH $model_path TOTAL_PARTS 8 DEVICES 0,1,2,3,4,5,6,7 BEAM_DEPTH 10 EXPAND_NUM_THOUGHT 3 BEAM_SIZE_THOUGHT 1 EXPAND_NUM_SEARCH 3 BEAM_SIZE_SEARCH 1 TEMPERATURE 0.7 LOG_PORT 6004 SEARCH_LOG_PORT 7001 POLICY_IP $ip POLICY_PORT 6002 ARM_IP none ARM_PORT 0 VALUE_IP $ip VALUE_PORT 7001 RETRIEVER_IP 22.5.226.141 RETRIEVER_PORT 5005 2>&1 &

nohup bash parallel_inference_72b_hierarchical.sh DATASET bamboogle MODEL_PATH $model_path TOTAL_PARTS 8 DEVICES 0,1,2,3,4,5,6,7 BEAM_DEPTH 10 EXPAND_NUM_THOUGHT 1 BEAM_SIZE_THOUGHT 1 EXPAND_NUM_SEARCH 1 BEAM_SIZE_SEARCH 1 TEMPERATURE 0.7 LOG_PORT 6005 SEARCH_LOG_PORT 7001 POLICY_IP $ip POLICY_PORT 6002 ARM_IP none ARM_PORT 0 VALUE_IP $ip VALUE_PORT 7001 RETRIEVER_IP 22.5.226.141 RETRIEVER_PORT 5005 2>&1 &

#######################################################################################################################################################################################################################################

