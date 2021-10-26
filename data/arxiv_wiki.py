import numpy as np
import pickle
from scipy.sparse import csc_matrix

def parse_arxiv_or_wiki(filename='CA-GrQc.txt'):
  """
  Parse either of 2 datasets:  
    Arxiv  GR-QC (General Relativity and Quantum Cosmology) data 
      Link: https://snap.stanford.edu/data/ca-GrQc.html
    Wiki vote network data
      Link: https://snap.stanford.edu/data/wiki-Vote.html
  into adjacency matrix 
  
  NodeID are not indexed from 0 or 1 so we have a separate dict to map them 
  
  Returns: adjacency matrix 

  """
  nodenames = set()
  node_map = {} # dictionary of idx => nodeID


  # popluate nodenames first
  with open(filename) as f:
    c = 0
    for line in f:
      c += 1
      if c >= 5:
        first, second = line.strip().split()
        nodenames.add(int(first))
        nodenames.add(int(second))

  n_nodes = len(nodenames)
  print("\nThere are %d nodes\n" % n_nodes)
  nodenames = list(nodenames)
  adj = np.zeros((n_nodes, n_nodes))

  # create a dictionary of node ID => idx
  for i in range(n_nodes):
    node_map[nodenames[i]] = i

  with open(filename) as f:
    c = 0
    for line in f:
      c += 1
      if c >= 5:
        first, second = line.strip().split()
        first_idx, second_idx = node_map[int(first)], node_map[int(second)]
        adj[first_idx][second_idx] = 1
        adj[second_idx][first_idx] = 1

  return adj


if __name__ == '__main__':

  # adj_arxiv = parse_arxiv_or_wiki(filename='CA-GrQc.txt')
  adj_arxiv = parse_arxiv_or_wiki(filename='CA-CondMat.txt')
  print(adj_arxiv)
  print((adj_arxiv == adj_arxiv.T).all())
  adj_arxiv3 = adj_arxiv @ adj_arxiv @ adj_arxiv
  adj_arxiv3_sparse = csc_matrix(adj_arxiv3)
  print(adj_arxiv3_sparse)
  with open('arxiv_cm3.pkl', 'wb') as f:
    pickle.dump(adj_arxiv3_sparse, f)
