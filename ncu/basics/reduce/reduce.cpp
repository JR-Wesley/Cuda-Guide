#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <iomanip>

// 1. Loop-based sum
template <typename T>
T loopSum(const std::vector<T>& data) {
    std::cout << "Step [Loop Sum]: ";
    for (const auto& x : data) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    return std::reduce(data.begin(), data.end(), T{0});
}

// 2. Fixed recursive reduce: operates on array of given size
template <typename T>
T recursiveReduce(T* data, int size) {
    std::cout << "Step [" << size << "]: ";
    for (int i = 0; i < size; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << "\n";

    // Base case: single element
    if (size == 1) {
        return data[0];
    }

    int stride = size / 2;

    // In-place reduction: first half += second half
    for (int i = 0; i < stride; ++i) {
        data[i] += data[i + stride];
    }

    // Recurse on the first half (size now is stride)
    return recursiveReduce(data, stride);
}

// Wrapper to avoid mutation and keep interface clean
template <typename T>
T recursiveReduceWrapper(std::vector<T> data) {
    assert(!data.empty() && (data.size() & (data.size() - 1)) == 0 &&
           "Size must be a power of 2");
    return recursiveReduce(data.data(), static_cast<int>(data.size()));
}

// Test function
template <typename T>
void testWithSize(int N) {
    std::cout << "\n=== Testing with N = " << N << " ===\n";

    std::vector<T> data(N);
    std::iota(data.begin(), data.end(), T{1});  // 1, 2, ..., N

    std::cout << "Original data: ";
    for (const auto& x : data) std::cout << x << " ";
    std::cout << "\n\n";

    auto sum1 = loopSum(data);
    auto sum2 = recursiveReduceWrapper(data);  // pass copy

    std::cout << "\nResults:\n";
    std::cout << "Loop Sum:           " << sum1 << "\n";
    std::cout << "Recursive Reduce:   " << sum2 << "\n";

    assert(sum1 == sum2);
    std::cout << "✅ Both results match!\n";
}

int main(int argc, char const *argv[]) {
    std::cout << std::fixed << std::setprecision(1);

    std::vector<int> sizes;
    sizes.reserve(argc - 1);
    for (int i = 1; i < argc; ++i) {
        sizes.push_back(std::stoi(argv[i]));
    }

    for (int sz : sizes) {
        testWithSize<int>(sz);
    }

    return 0;
}