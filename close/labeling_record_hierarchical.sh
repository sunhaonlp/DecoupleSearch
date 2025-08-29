


####################################################################################################################################################################################
## 1.7
nohup /mnt/workspace/sunhao/miniconda3/envs/myenv/bin/python /mnt/workspace/sunhao/code/myenv-rag/statistic_server.py --port 6000 --identifier overall 2>&1 > server.log &


bash data_labeling_paper.sh paper_labeling_data_v1 20 qwen-plus-latest 5 40 100 1 1_14_13_qwen_fusion_all 0 9 6000

bash data_labeling_paper.sh paper_labeling_data_v1 20 qwen-max-latest 5 40 100 1 1_14_13_qwen_fusion_all 10 19 6000


####################################################################################################################################################################################
## 1.25
nohup /mnt/workspace/sunhao/miniconda3/envs/myenv/bin/python /mnt/workspace/sunhao/code/myenv-rag/statistic_server.py --port 6000 --identifier paper_1_25 2>&1 > server.log &

bash data_labeling_paper.sh paper_labeling_data_v1 100 qwen-max-latest 5 40 100 1 None 0 49 6000 22.8.154.146 5001

bash data_labeling_paper.sh paper_labeling_data_v1 100 qwen-max-latest 5 40 100 1 None 50 99 6000 22.8.154.146 5002





CUDA_VISIBLE_DEVICES=0  /mnt/workspace/sunhao/miniconda3/envs/myenv/bin/python retriever_server.py --port 5001 2>&1 &
CUDA_VISIBLE_DEVICES=1  /mnt/workspace/sunhao/miniconda3/envs/myenv/bin/python retriever_server.py --port 5002 2>&1 &
CUDA_VISIBLE_DEVICES=2  /mnt/workspace/sunhao/miniconda3/envs/myenv/bin/python retriever_server_2.py --port 5003 2>&1 &
CUDA_VISIBLE_DEVICES=3  /mnt/workspace/sunhao/miniconda3/envs/myenv/bin/python retriever_server_2.py --port 5004 2>&1 &
CUDA_VISIBLE_DEVICES=4  /mnt/workspace/sunhao/miniconda3/envs/myenv/bin/python retriever_server_2.py --port 5005 2>&1 &
CUDA_VISIBLE_DEVICES=5  /mnt/workspace/sunhao/miniconda3/envs/myenv/bin/python retriever_server_2.py --port 5006 2>&1 &
CUDA_VISIBLE_DEVICES=6  /mnt/workspace/sunhao/miniconda3/envs/myenv/bin/python retriever_server_2.py --port 5007 2>&1 &
CUDA_VISIBLE_DEVICES=7  /mnt/workspace/sunhao/miniconda3/envs/myenv/bin/python retriever_server_2.py --port 5008 2>&1 &



####################################################################################################################################################################################
## 1.29


#bash data_labeling_paper.sh paper_labeling_data_v1 100 qwen-plus-latest 5 20 3 1 None 20 29 6000 22.3.232.32 5001 0
#bash data_labeling_paper.sh paper_labeling_data_v1 100 qwen-plus-latest 5 20 3 1 None 30 39 6000 22.3.232.32 5002 1
#
#
#bash data_labeling_paper.sh paper_labeling_data_v1 100 qwen-plus-0919 5 20 3 1 None 40 49 6000 22.3.232.32 5003 0
#bash data_labeling_paper.sh paper_labeling_data_v1 100 qwen-plus-0919 5 20 3 1 None 50 59 6000 22.3.232.32 5004 1
#bash data_labeling_paper.sh paper_labeling_data_v1 100 qwen-plus-0919 5 20 3 1 None 60 69 6000 22.3.232.32 5005 0
#bash data_labeling_paper.sh paper_labeling_data_v1 100 qwen-plus-0919 5 20 3 1 None 70 79 6000 22.3.232.32 5006 1


#bash data_labeling_paper.sh paper_labeling_data_v1 100 qwen-max-0919 5 20 3 1 None 80 89 6000 22.3.232.32 5007 0
#bash data_labeling_paper.sh paper_labeling_data_v1 100 qwen-max-0919 5 20 3 1 None 90 99 6000 22.3.232.32 5008 1

nohup /mnt/workspace/sunhao/miniconda3/envs/myenv/bin/python statistic_server.py --port 6003 --identifier hierar_2_4 2>&1 > server.log &




bash data_labeling_hierarchical.sh paper_labeling_data_v2 120 qwen-plus-latest 5 20 3 1 1_30_22_qwen_fusion_all 0 9 6003 'None' 0 0 'None' 0
bash data_labeling_hierarchical.sh paper_labeling_data_v2 120 qwen-plus-latest 5 20 3 1 1_30_22_qwen_fusion_all 10 19 6003 'None' 0 1 'None' 0

bash data_labeling_hierarchical.sh paper_labeling_data_v2 120 qwen-plus-0919 5 20 3 1 1_30_22_qwen_fusion_all 20 29 6003 'None' 0 0 'None' 0
bash data_labeling_hierarchical.sh paper_labeling_data_v2 120 qwen-plus-0919 5 20 3 1 1_30_22_qwen_fusion_all 30 39 6003 'None' 0 1 'None' 0


bash data_labeling_hierarchical.sh paper_labeling_data_v2 120 qwen-max-latest 5 20 3 1 1_30_22_qwen_fusion_all 40 49 6003 'None' 0 0 'None' 0
bash data_labeling_hierarchical.sh paper_labeling_data_v2 120 qwen-max-latest 5 20 3 1 1_30_22_qwen_fusion_all 50 59 6003 'None' 0 1 'None' 0

bash data_labeling_hierarchical.sh paper_labeling_data_v2 120 qwen-max-0919 5 20 3 1 1_30_22_qwen_fusion_all 60 69 6003 'None' 0 0 'None' 0
bash data_labeling_hierarchical.sh paper_labeling_data_v2 120 qwen-max-0919 5 20 3 1 1_30_22_qwen_fusion_all 70 79 6003 'None' 0 1 'None' 0


bash data_labeling_hierarchical.sh paper_labeling_data_v2 120 local_qwen 5 20 3 1 1_30_22_qwen_fusion_all 80 99 6003 'None' 0 0 22.3.232.32 6001

bash data_labeling_hierarchical.sh paper_labeling_data_v2 120 local_qwen 5 20 3 1 1_30_22_qwen_fusion_all 100 119 6003 'None' 0 1 22.6.224.114 6001


