#include <iostream>
#include <omp.h>
#include <climits> // For INT_MIN and INT_MAX
#include <chrono>  // For execution time measurement

using namespace std;

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    int* arr = new int[n];

    // Input array elements
    cout << "Enter the elements:\n";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    // Initialize variables for reduction
    int minVal = INT_MAX;
    int maxVal = INT_MIN;
    long long sum = 0; // Use long long for larger sums

    // Start measuring execution time
    auto start = chrono::high_resolution_clock::now();

    // Parallel reduction using OpenMP
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(+:sum)
    for (int i = 0; i < n; i++) {
        if (arr[i] < minVal) minVal = arr[i]; // Find min
        if (arr[i] > maxVal) maxVal = arr[i]; // Find max
        sum += arr[i]; // Calculate sum
    }

    // End measuring execution time
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    // Calculate average
    double average = static_cast<double>(sum) / n;

    // Display results
    cout << "\nResults of Parallel Reduction:\n";
    cout << "Minimum Value: " << minVal << endl;
    cout << "Maximum Value: " << maxVal << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << average << endl;

    // Display execution time
    cout << "Execution Time: " << duration.count() << " microseconds" << endl;

    // Free dynamically allocated memory
    delete[] arr;

    return 0;
}