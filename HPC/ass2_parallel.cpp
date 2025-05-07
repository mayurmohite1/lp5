#include <iostream>
#include <vector>
#include <omp.h>
#include <ctime>
using namespace std;
void bubbleSortParallel(vector<int>& arr) {
int n = arr.size();
bool swapped = true;
while (swapped) {
swapped = false;
#pragma omp parallel for
for (int i = 0; i < n - 1; i++) {
if (arr[i] > arr[i + 1]) {
swap(arr[i], arr[i + 1]);
swapped = true;
}
}
}
}
void mergeParallel(vector<int>& arr, int l, int m, int r) {
vector<int> temp;
int left = l;
int right = m + 1;
while (left <= m && right <= r) {
if (arr[left] <= arr[right]) {
temp.push_back(arr[left]);
left++;
}
else {
temp.push_back(arr[right]);
right++;
}
}
while (left <= m) {
temp.push_back(arr[left]);
left++;
}
while (right <= r) {
temp.push_back(arr[right]);
right++;
}
for (int i = l; i <= r; i++) {
arr[i] = temp[i - l];
}
}
void mergeSortParallel(vector<int>& arr, int l, int r) {
if (l < r) {
int m = l + (r - l) / 2;
#pragma omp parallel sections
{
#pragma omp section
mergeSortParallel(arr, l, m);
#pragma omp section
mergeSortParallel(arr, m + 1, r);
}
mergeParallel(arr, l, m, r);
}
}
int main() {
int n;
cout << "Enter the number of elements: ";
cin >> n;
vector<int> arr(n);
cout << "Enter the elements: ";
for (int i = 0; i < n; i++)
cin >> arr[i];
// Parallel Bubble Sort
vector<int> arr1 = arr; // Copy array for Bubble Sort
clock_t bubbleStart = clock();
bubbleSortParallel(arr1);
clock_t bubbleEnd = clock();
cout << "Sorted array using Bubble Sort (Parallel): ";
for (int num : arr1)
cout << num << " ";
cout << endl;
// Parallel Merge Sort
vector<int> arr2 = arr; // Copy array for Merge Sort
clock_t mergeStart = clock();
mergeSortParallel(arr2, 0, n - 1);
clock_t mergeEnd = clock();
cout << "Sorted array using Merge Sort (Parallel): ";
for (int num : arr2)
cout << num << " ";
cout << endl;
// Time calculation in milliseconds
double bubbleDuration = double(bubbleEnd - bubbleStart) / CLOCKS_PER_SEC * 1000; //

double mergeDuration = double(mergeEnd - mergeStart) / CLOCKS_PER_SEC * 1000; //

cout << "Parallel Bubble sort time in milliseconds: " << bubbleDuration << " ms" << endl;
cout << "Parallel Merge sort time in milliseconds: " << mergeDuration << " ms" << endl;
return 0;
}