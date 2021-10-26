import numpy as np
import pickle

def parse_roget(filename='Roget.net'):
  """
  Parse Roget network data into adjacency matrix 
  Refer to Graph Estrada Index experiment in Hutch++ (https://arxiv.org/pdf/2010.09649.pdf) 
    (Section 5.2) 
  
  Args:
    filename: path to data file, default is Roget.net  

  Returns: node name dictionary and adjacency square, binary matrix 

  """
  nodes = {}
  adj = np.zeros((1022, 1022))

  with open(filename) as f:
    c = 0
    for line in f:
      c += 1
      if 2 <= c <= 1023:
        line = line.replace('"', '')
        fields = line.split()
        idx = fields[0]
        words = ' '.join(fields[1:])
        # print(idx, words)
        nodes[idx] = words
      elif c >= 1025:
        # adjacency matrix
        idxes = [int(x) for x in line.strip().split()]
        current_node = c - 1025
        for idx in idxes:
          adj[current_node][idx-1] = 1
          adj[idx-1][current_node] = 1

  return nodes, adj


if __name__ == '__main__':
  nodes, adj = parse_roget()
  # print(adj[0])
  print(len(np.where(adj > 0)[0]))
  print((adj == adj.T).all())
  with open('roget.pkl', 'wb') as f:
    pickle.dump(adj, f)
