import random
import math
import os

def calculate_distance(p1, p2):
    """
    Calculates Euclidean distance and uses CEILING to ensure 
    triangle inequality holds for integers.
    """
    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    # math.ceil is crucial here. 
    # If side C <= A + B, then ceil(C) <= ceil(A) + ceil(B).
    return math.ceil(dist)

def check_triangle_inequality(num_nodes, adj_matrix):
    """
    Verifies that the generated graph satisfies triangle inequality 
    for ALL triplets. Returns True if valid.
    """
    for i in range(num_nodes):
        for j in range(num_nodes):
            for k in range(num_nodes):
                # Check: d(i,j) <= d(i,k) + d(k,j)
                dist_ij = adj_matrix[i][j]
                dist_ik = adj_matrix[i][k]
                dist_kj = adj_matrix[k][j]
                
                if dist_ij > dist_ik + dist_kj:
                    print(f"VIOLATION: {i}->{j} ({dist_ij}) > {i}->{k} ({dist_ik}) + {k}->{j} ({dist_kj})")
                    return False
    return True

def generate_input_file(filename, alpha, num_nodes, num_friends):
    # 1. Generate Distinct Coordinates
    # Using a larger grid (0-10000) reduces the chance of duplicate coordinates
    coords = []
    seen = set()
    while len(coords) < num_nodes:
        x = random.randint(0, 10000)
        y = random.randint(0, 10000)
        if (x,y) not in seen:
            seen.add((x,y))
            coords.append((x, y))

    # 2. Build Adjacency Matrix (for easy checking) and List
    # We use a matrix first to easily check the inequality
    adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                adj_matrix[i][j] = 0
            else:
                weight = calculate_distance(coords[i], coords[j])
                # Ensure weight is at least 1 for distinct nodes
                adj_matrix[i][j] = max(1, weight)

    # 3. Verify Triangle Inequality
    # This runs in O(N^3), which is trivial for N=40 (64,000 checks)
    if not check_triangle_inequality(num_nodes, adj_matrix):
        print(f"Error: Failed to generate valid graph for {filename}")
        return

    # 4. Write to File
    with open(filename, 'w') as f:
        # Line 1: alpha
        f.write(f"{alpha:.5g}\n")
        
        # Line 2: |V| |H|
        f.write(f"{num_nodes} {num_friends}\n")
        
        # Line 3: List of home nodes (Sample distinct friends excluding node 0)
        available_indices = list(range(1, num_nodes))
        homes = random.sample(available_indices, num_friends)
        homes.sort()
        f.write(" ".join(map(str, homes)) + "\n")
        
        # Line 4+: Adjacency List
        for i in range(num_nodes):
            # Degree is N-1 for a complete graph
            degree = num_nodes - 1
            f.write(f"{i} {degree}\n")
            for j in range(num_nodes):
                if i == j: continue
                weight = adj_matrix[i][j]
                f.write(f"{j} {weight}\n")
    
    print(f"Generated {filename} successfully (Verified Triangle Inequality).")

def main():
    # File configurations based on [cite: 83-87] and folder structure [cite: 281-284]
    configs = [
        ("20_03.in", 0.3, 20, 10),
        ("20_10.in", 1.0, 20, 10),
        ("40_03.in", 0.3, 40, 20),
        ("40_10.in", 1.0, 40, 20)
    ]
    
    if not os.path.exists('inputs'):
        os.makedirs('inputs')

    for config in configs:
        generate_input_file(os.path.join('inputs', config[0]), *config[1:])

if __name__ == "__main__":
    main()