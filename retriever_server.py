import time
from flask import Flask, request, jsonify
import faiss
from pyserini.search.lucene import LuceneSearcher
from pyserini.encode import DprQueryEncoder
from threading import Lock
import argparse
import logging

app = Flask(__name__)

# 设置日志级别为DEBUG
logging.basicConfig(level=logging.DEBUG)

# 创建一个锁对象
lock = Lock()

# GPU 索引和资源的初始化
index_path = 'retriever/faiss.wikipedia-dpr-100w.dpr_multi.20200127.f403c3/index'
gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss.read_index(index_path))

docid_path = 'retriever/faiss.wikipedia-dpr-100w.dpr_multi.20200127.f403c3/docid'
with open(docid_path, 'r') as f:
    docids = f.read().splitlines()

encoder = DprQueryEncoder("retriever/dpr-question_encoder-multiset-base", device='cuda:0')
searcher_sparse = LuceneSearcher('retriever/lucene-index.wikipedia-dpr-100w.20210120.d1b9e6')


def search_index(query, top_k=10):
    with lock:
        query_embedding = encoder.encode(query)
        distances, indices = gpu_index.search(query_embedding.reshape(1, -1), 10)

    ret = []
    for i in range(min(top_k, len(indices[0]))):
        docid = indices[0, i]
        doc_data = searcher_sparse.doc(docids[docid])
        if doc_data:
            ret.append({'id': str(docid), 'text': eval(doc_data.raw())['contents']})
    return ret


@app.route('/retrieve', methods=['POST'])
def retrieve():
    try:
        data = request.json
        app.logger.debug(f"Received data: {data}")

        query = data['query']
        top_k = data.get('top_k', 5)

        results = search_index(query, top_k)

        app.logger.debug(f"Search results: {results}")
        return jsonify(results)

    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({'error': 'An error occurred during processing'}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行检索服务器')
    parser.add_argument('--port', type=int, default=5000, help='指定服务器运行的端口（默认：5000）')
    args = parser.parse_args()

    # 使用指定的端口运行应用
    app.run(host='0.0.0.0', port=args.port, threaded=False)
