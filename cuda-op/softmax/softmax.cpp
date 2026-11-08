#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <random>
#include <limits>
#include <functional>
#include <numeric>

std::vector<float> naive_softmax(const std::vector<float>& x) {
    // 计算归一化项
    float d_V = std::accumulate(x.begin(), x.end(), 0.0f, 
        [](float sum, float val) { return sum + std::exp(val); });
    
    // 计算输出值
    std::vector<float> y(x.size());
    std::transform(x.begin(), x.end(), y.begin(), 
        [d_V](float val) { return std::exp(val) / d_V; });
    
    return y;
}

std::vector<float> safe_softmax(const std::vector<float>& x) {
    // 计算最大值
    float max_val = *std::max_element(x.begin(), x.end());
    
    // 计算归一化项
    float d_V = std::accumulate(x.begin(), x.end(), 0.0f, 
        [max_val](float sum, float val) { return sum + std::exp(val - max_val); });
    
    // 计算输出值
    std::vector<float> y(x.size());
    std::transform(x.begin(), x.end(), y.begin(), 
        [max_val, d_V](float val) { return std::exp(val - max_val) / d_V; });
    
    return y;
}

std::vector<float> online_softmax(const std::vector<float>& x) {
    // 使用 std::accumulate 同时计算最大值和归一化项
    auto result = std::accumulate(x.begin(), x.end(), 
        std::make_pair(-std::numeric_limits<float>::infinity(), 0.0f),
        [](const std::pair<float, float>& acc, float val) {
            float old_m = acc.first;
            float d = acc.second;
            float m = std::max(old_m, val);
            
            // 调整归一化项以适应新的最大值
            float new_d = (m != old_m) ? d * std::exp(old_m - m) : d;
            // 添加新值到归一化项
            new_d += std::exp(val - m);
            
            return std::make_pair(m, new_d);
        });
    
    float max = result.first;
    float norm = result.second;
    
    // 计算输出值
    std::vector<float> y(x.size());
    std::transform(x.begin(), x.end(), y.begin(), 
        [max, norm](float val) { return std::exp(val - max) / norm; });
    
    return y;
}

// 生成测试数据
std::vector<float> generate_test_data(int size) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    
    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
    
    return data;
}

// 打印结果的前几个元素
void print_first_elements(const std::vector<float>& vec, int count) {
    for (int i = 0; i < std::min(count, (int)vec.size()); i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

// 使用函数指针测试 softmax 实现
void test_softmax_implementation(
    const std::function<std::vector<float>(const std::vector<float>&)>& softmax_func,
    const std::vector<float>& test_data,
    const std::string& name,
    int iterations) {
    
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> result;
    for (int i = 0; i < iterations; i++) {
        result = softmax_func(test_data);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << name << ":" << std::endl;
    std::cout << "  Result (first 10 elements): ";
    print_first_elements(result, 10);
    std::cout << "  Time for " << iterations << " iterations: " << duration.count() << " microseconds" << std::endl;
    std::cout << "  Average time per iteration: " << duration.count() / (float)iterations << " microseconds" << std::endl;
}

// 设置测试数据大小
static constexpr int data_size = 1000000;  // 1M elements
static constexpr int iterations = 100;     // 运行100次以获得更准确的时间测量

int main(int argc, char *argv[]) {
    // 生成测试数据
    std::vector<float> test_data = generate_test_data(data_size);
    
    // 使用函数指针测试所有实现
    test_softmax_implementation(naive_softmax, test_data, "Naive Softmax", iterations);
    std::cout << std::endl;
    test_softmax_implementation(safe_softmax, test_data, "Safe Softmax", iterations);
    std::cout << std::endl;
    test_softmax_implementation(online_softmax, test_data, "Online Softmax", iterations);
    
    return 0;
}