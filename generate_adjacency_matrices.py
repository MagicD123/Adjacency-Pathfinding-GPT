import random
import numpy as np
import json
from typing import List, Dict

def generate_connected_adjacency_matrix(num_nodes: int, max_degree: int) -> np.ndarray:
    """
    Generate a connected adjacency matrix with each node having a degree not exceeding max_degree.
    """
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    degrees = [0] * num_nodes
    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    
    # Create a random spanning tree to ensure connectivity
    for i in range(num_nodes - 1):
        a, b = nodes[i], nodes[i + 1]
        adjacency_matrix[a][b] = adjacency_matrix[b][a] = 1
        degrees[a] += 1
        degrees[b] += 1
    
    # Try adding extra edges without exceeding max_degree
    possible_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    random.shuffle(possible_edges)
    
    for (i, j) in possible_edges:
        if adjacency_matrix[i][j] == 0 and degrees[i] < max_degree and degrees[j] < max_degree:
            adjacency_matrix[i][j] = adjacency_matrix[j][i] = 1
            degrees[i] += 1
            degrees[j] += 1
    
    return adjacency_matrix

def generate_network_adjacency_matrices(N: int) -> List[Dict]:
    """
    Generate N adjacency matrices.
    """
    matrices = []
    for matrix_id in range(1, N + 1):
        num_nodes = random.randint(5, 10)
        max_degree = num_nodes // 2
        adjacency_matrix = generate_connected_adjacency_matrix(num_nodes, max_degree)
        
        matrices.append({
            "matrix_id": matrix_id,
            "size": num_nodes,
            "adjacency_matrix": adjacency_matrix.tolist()
        })
    
    return matrices

def save_to_json(data: List[Dict], filename: str):
    """
    Save generated data to a JSON file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {filename}")

def main():
    N = int(input("Enter the number of adjacency matrices to generate: "))
    matrices = generate_network_adjacency_matrices(N)
    save_to_json(matrices, "adjacency_matrices.json")

if __name__ == "__main__":
    main()
