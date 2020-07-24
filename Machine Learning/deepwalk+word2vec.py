
import pandas as pd
import os
import networkx as nx
from gensim.models import Word2Vec
from joblib import Parallel, delayed
import itertools
import random
import numpy as np


#Deepwalk，返回值N个节点*M路径长度的路径矩阵
class RandomWalker:
    def __init__(self, G):
        self.G = G

    def partition_num(self,num, workers):
        if num % workers == 0:
            return [num // workers] * workers
        else:
            return [num // workers] * workers + [num % workers]

    #deepwalk_walk为单个节点行走
    def deepwalk_walk(self, walk_length, start_node):

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    #_simulate_walks是deepwalk_walk的外层循环，对给定的所有节点调用deepwalk_walk行走
    def _simulate_walks(self, nodes, num_walks, walk_length, ):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            print("simulating walks...", _)
            for v in nodes:
                walks.append(self.deepwalk_walk(walk_length, start_node=v))
        print("simulation done!")
        return walks

    #配置并行处理
    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):
        G = self.G
        nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            self.partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks


#RandomWalk + Word2vec 返回值为embedding稠密矩阵
class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}
        
        self.walker = RandomWalker(graph)
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)


    def train(self, embed_size=5, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings




if __name__ == "__main__":

	G = nx.read_edgelist('../edgelist/test.txt', nodetype=None)
	model = DeepWalk(G, walk_length=10, num_walks=10, workers=2)
	model.train(window_size=5, iter=3)
	embeddings = model.get_embeddings()
	np.save('embeddings.npy',embeddings)
    