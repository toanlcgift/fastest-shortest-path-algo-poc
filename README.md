### Breaking the Sorting Barrier for Directed Single-Source Shortest Paths
----

### Result

- Dijkstra → Complete, accurate, but slower.
- BMSSP → Much faster but only covers a fraction of the graph and produces large distance errors (max diff ~85).
- Trade-off: speed vs accuracy.

```python
Generating graph: n=200000, m=800000, seed=0
Graph generated. avg out-degree ≈ 4.000
Dijkstra: time=1.714210s, relaxations=800000, heap_ops=533360, reachable=200000
BMSSP params: top-level l=2
BMSSP: time=0.079001s, relaxations=26373, reachable=8629, B'=0.0, |U_final|=2048
Distance agreement (max abs diff on commonly reachable nodes): 8.540939e+01
```

| **Metric**                | **Dijkstra** | **BMSSP**        | **Notes**                                 |       |                              |
| ------------------------- | ------------ | ---------------- | ----------------------------------------- | ----- | ---------------------------- |
| **Nodes (n)**             | 200,000      | 200,000          | Same graph for both algorithms            |       |                              |
| **Edges (m)**             | 800,000      | 800,000          | Avg out-degree ≈ 4                        |       |                              |
| **Execution Time (s)**    | 1.7142       | 0.0790           | BMSSP is \~22× faster                     |       |                              |
| **Relaxations**           | 800,000      | 26,373           | BMSSP examines \~3% of edges              |       |                              |
| **Heap Ops**              | 533,360      | —                | BMSSP doesn’t use standard PQ for all ops |       |                              |
| **Reachable Nodes**       | 200,000      | 8,629            | BMSSP covers only \~4.3% of nodes         |       |                              |
| **B′ (bound)**            | —            | 0.0              | Indicates aggressive pruning              |       |                              |
| \*\*                      | U\_final     | (unresolved)\*\* | —                                         | 2,048 | Nodes BMSSP left unprocessed |
| **Max Abs Distance Diff** | 0            | 85.41            | Large error on overlapping reachable set  |       |                              |

[alphaxiv](https://papers-pdfs.assets.alphaxiv.org/2504.17033v2.pdf)


# POC Dotnet Project  

## Overview  
This project demonstrates a .NET-based implementation of a graph processing application. It includes a CLI interface for configuring the number of nodes, edges, and a random seed for generating graph data.  

## Features  
- Command-line interface for input configuration.  
- Efficient graph processing using .NET 9 features.  
- Supports large-scale graph data with customizable parameters.  

## Requirements  
- .NET 9 SDK  
- C# 13.0  

## Usage  

### Build the Project  
1. Clone the repository.  
2. Open the project in Visual Studio.  
3. Build the solution.  

### Run the Application  
Use the following command to run the application:
### Example
### Parameters  
- `-n` or `--nodes`: Number of nodes in the graph (default: 200,000).  
- `-m` or `--edges`: Number of edges in the graph (default: 800,000).  
- `-s` or `--seed`: Random seed for graph generation (default: 0).  

### C# Output => <b> running faster than python</b>
```shell
Generating graph: n=200000, m=800000, seed=0
Graph generated. avg out-degree ≈ 4.000
Dijkstra: time=0.356301s, relaxations=800000, heap_ops=533202, reachable=200000
BMSSP params: top-level l=2
BMSSP: time=0.054503s, relaxations=26910, reachable=8788, B'=0, |U_final|=2048
Distance agreement (max abs diff on commonly reachable nodes): 8.073533E+001
```
