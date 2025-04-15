#include <iostream>
#include <vector>
#include <queue>
#include <stdexcept> // For exception handling like out_of_range

using namespace std;

// Function to perform Breadth-First Search (BFS) from a starting node
void bfs(int startNode, const vector<vector<int>>& adjList) {
    // Check for invalid starting node
    if (startNode < 0 || startNode >= adjList.size()) {
        throw out_of_range("Start node is out of range");
    }

    // Track visited nodes to avoid revisiting
    vector<bool> visited(adjList.size(), false);

    // Queue for BFS traversal (FIFO)
    queue<int> q;

    // Start from the given node
    visited[startNode] = true;  // Mark start node as visited
    q.push(startNode);          // Push it to the queue

    // Process the queue until it's empty
    while (!q.empty()) {
        int currentNode = q.front(); // Get front node from the queue
        q.pop();                     // Remove it from the queue

        cout << currentNode << " ";  // Print the visited node

        // Explore all unvisited neighbors of the current node
        for (int neighbor : adjList[currentNode]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true; // Mark neighbor as visited
                q.push(neighbor);         // Add neighbor to the queue
            }
        }
    }
}

int main() {
    int n, m; // n = number of nodes, m = number of edges

    // Take input for number of nodes and edges
    cout << "Enter the number of nodes and edges: ";
    cin >> n >> m;

    // Validate input
    if (n <= 0 || m < 0) {
        cout << "Invalid input. Number of nodes must be positive, and number of edges must be non-negative." << endl;
        return 1;
    }

    // Create an adjacency list to represent the graph
    vector<vector<int>> adjList(n);

    // Input edges (u v pairs)
    cout << "Enter the edges (u v):" << endl;
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;

        // Validate each edge's nodes
        if (u < 0 || u >= n || v < 0 || v >= n) {
            cout << "Invalid edge input: nodes must be in the range [0, " << n - 1 << "]" << endl;
            return 1;
        }

        // Add edge to adjacency list (undirected graph)
        adjList[u].push_back(v);
        adjList[v].push_back(u);
    }

    // Input the starting node for BFS
    int startNode;
    cout << "Enter the start node: ";
    cin >> startNode;

    // Validate the starting node
    if (startNode < 0 || startNode >= n) {
        cout << "Start node is out of range!" << endl;
        return 1;
    }

    // Perform BFS from the given node
    cout << "BFS traversal starting from node " << startNode << ": ";
    bfs(startNode, adjList);
    cout << endl;

    return 0;
}
