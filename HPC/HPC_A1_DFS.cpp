#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>
#include <stdexcept> // For exception handling like out_of_range

using namespace std;

// Function to perform parallel DFS traversal using OpenMP
void dfs_parallel(int startNode, const vector<vector<int>>& adjList) {
    // Check if the starting node is valid
    if (startNode < 0 || startNode >= adjList.size()) {
        throw out_of_range("Start node is out of range");
    }

    // Initialize a visited array to track which nodes have been visited
    vector<bool> visited(adjList.size(), false);

    // Shared stack to manage the DFS traversal
    stack<int> s;

    // Begin parallel region
    #pragma omp parallel
    {
        // Only one thread should perform the initial push to the stack
        #pragma omp single
        {
            visited[startNode] = true;     // Mark the starting node as visited
            s.push(startNode);             // Push the start node onto the stack
            cout << startNode << " ";      // Print the visited node
        }

        // Begin loop to traverse graph while stack is not empty
        while (true) {
            int currentNode = -1; // Initialize with an invalid value

            // Ensure only one thread accesses/modifies the stack at a time
            #pragma omp critical
            {
                if (!s.empty()) {
                    currentNode = s.top(); // Get the top node from the stack
                    s.pop();               // Remove it from the stack
                }
            }

            // If no node was available to pop, then all are processed — exit loop
            if (currentNode == -1)
                break;

            // Explore all neighbors of the current node in parallel
            #pragma omp parallel for
            for (int i = 0; i < adjList[currentNode].size(); ++i) {
                int neighbor = adjList[currentNode][i]; // Get neighbor node

                // If neighbor is not visited
                if (!visited[neighbor]) {
                    // Synchronize access to shared `visited` and `stack`
                    #pragma omp critical
                    {
                        // Double-check to avoid race conditions
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;  // Mark neighbor as visited
                            s.push(neighbor);          // Push neighbor to stack
                            cout << neighbor << " ";   // Print the visited neighbor
                        }
                    }
                }
            }
        }
    }
}

int main() {
    int n, m; // n = number of nodes, m = number of edges

    // Input number of nodes and edges
    cout << "Enter the number of nodes and edges: ";
    cin >> n >> m;

    // Validate input
    if (n <= 0 || m < 0) {
        cout << "Invalid input. Number of nodes must be positive, and number of edges must be non-negative." << endl;
        return 1;
    }

    // Create adjacency list to represent the graph
    vector<vector<int>> adjList(n);

    // Input the edges
    cout << "Enter the edges (u v):" << endl;
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;

        // Validate edge nodes
        if (u < 0 || u >= n || v < 0 || v >= n) {
            cout << "Invalid edge input: nodes must be in the range [0, " << n - 1 << "]" << endl;
            return 1;
        }

        // Add edge to the adjacency list (undirected graph)
        adjList[u].push_back(v);
        adjList[v].push_back(u);
    }

    // Input the starting node for DFS
    int startNode;
    cout << "Enter the start node: ";
    cin >> startNode;

    // Validate the start node
    if (startNode < 0 || startNode >= n) {
        cout << "Start node is out of range!" << endl;
        return 1;
    }

    // Perform DFS traversal
    cout << "DFS traversal starting from node " << startNode << ": ";
    dfs_parallel(startNode, adjList);
    cout << endl;

    return 0;
}