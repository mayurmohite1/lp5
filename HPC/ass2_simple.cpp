#include <vector>
#include <ctime>
using namespace std;
void bubbleSortSequential(vector<int>& arr) {
int n = arr.size();
bool swapped = true;
while (swapped) {
swapped = false;
for (int i = 0; i < n - 1; i++) {
if (arr[i] > arr[i + 1]) {
swap(arr[i], arr[i + 1]);
swapped = true;
}
}
}
}
void mergeSequential(vector<int>& arr, int l, int m, int r) {
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
void mergeSortSequential(vector<int>& arr, int l, int r) {
if (l < r) {
int m = l + (r - l) / 2;
mergeSortSequential(arr, l, m);
mergeSortSequential(arr, m + 1, r);
mergeSequential(arr, l, m, r);
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
// Sequential Bubble Sort
vector<int> arr1 = arr; // Copy array for Bubble Sort
clock_t bubbleStart = clock();
bubbleSortSequential(arr1);
clock_t bubbleEnd = clock();
cout << "Sorted array using Bubble Sort (Sequential): ";
for (int num : arr1)
cout << num << " ";
cout << endl;
// Sequential Merge Sort
vector<int> arr2 = arr; // Copy array for Merge Sort
clock_t mergeStart = clock();
mergeSortSequential(arr2, 0, n - 1);
clock_t mergeEnd = clock();
cout << "Sorted array using Merge Sort (Sequential): ";
for (int num : arr2)
cout << num << " ";
cout << endl;
// Time calculation (in milliseconds)
double bubbleDuration = double(bubbleEnd - bubbleStart) / CLOCKS_PER_SEC * 1000; //
Convert to milliseconds
double mergeDuration = double(mergeEnd - mergeStart) / CLOCKS_PER_SEC * 1000; //
Convert to milliseconds
cout << "Sequential Bubble sort time in milliseconds: " << bubbleDuration << " ms" << endl;
cout << "Sequential Merge sort time in milliseconds: " << mergeDuration << " ms" << endl;
return 0;
}