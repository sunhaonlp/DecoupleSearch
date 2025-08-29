from flask import Flask, request, jsonify
import pickle
from threading import Lock, Thread
import argparse
import glob
import os
import wandb
from datetime import datetime
import time
from utils import *
import torch

app = Flask(__name__)
lock = Lock()

# 全局变量占位符
custom_model = None
tokenizer = None

def setup_model_and_tokenizer(args):
    model, tokenizer = load_hf_lm_and_tokenizer(
        model_name_or_path=args.model_path,
        tokenizer_name_or_path=args.model_path,
        device_index=args.base_device,
        load_in_8bit=args.load_in_8bit,
        load_in_half=True,
        device_map={'': f'cuda:{args.base_device}'}
    )
    custom_model = CustomModel_hierarchical(model, args.model_path + '/value_head_thought.safetensors',  args.model_path + '/value_head_search.safetensors')
    custom_model.to(f'cuda:{args.base_device}')
    custom_model.eval()
    return custom_model, tokenizer

@app.route('/statistic', methods=['POST'])
def retrieve():
    global custom_model, tokenizer

    # 之前可能会从request中获取数据，目前假设`messages`和`functions`已经定义
    input_data = request.get_json()
    input_ids = input_data.get('input_ids')
    input_ids = torch.tensor([input_ids]).to(custom_model.model.device)
    with torch.no_grad():
        outputs = custom_model(input_ids=input_ids, attention_mask = torch.ones(input_ids.shape),return_dict=True)
        values_thought = outputs["values_thought"][:, -1].cpu().tolist()
        values_search = outputs["values_search"][:, -1].cpu().tolist()

    return jsonify(values_thought=values_thought, values_search=values_search)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the retrieval server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--identifier', type=str, default='qwen', help='Identifier for the server')
    parser.add_argument('--model_path', type=str, required=True, help='Model path')
    parser.add_argument('--base_device', type=int, default=0, help='CUDA device index')
    parser.add_argument('--load_in_8bit', action='store_true', help='Load model in 8bit precision')

    args = parser.parse_args()

    # 初始化模型和tokenizer，赋值给全局变量
    custom_model, tokenizer = setup_model_and_tokenizer(args)

    # 运行应用
    app.run(host='0.0.0.0', port=args.port, threaded=True)
