#include <iostream>
#include <vector>
#include <memory>
#include <chrono> // For timing

// Required for Windows memory usage functions
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include "engine.h"
#include "neurals.h"
#endif


template<typename T>
std::vector<std::shared_ptr<Value<T>>> convert_to_values(const std::vector<T>& input) {
    std::vector<std::shared_ptr<Value<T>>> result;
    result.reserve(input.size());
    for (const auto& val : input) {
        result.push_back(std::make_shared<Value<T>>(val));
    }
    return result;
}



size_t get_current_memory_usage_kb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        // WorkingSetSize is the amount of physical memory currently assigned to the process
        return pmc.WorkingSetSize / 1024;
    }
    return 0;
#else
    // For non-Windows systems, this function does nothing.
    // You would add implementations for Linux/macOS here.
    std::cout << "Memory usage reporting is only implemented for Windows in this example." << std::endl;
    return 0;
#endif
}


int main() {
    std::cout << "Starting training loop with profiling..." << std::endl;

    // --- 1. PROFILING SETUP ---
    size_t mem_before = get_current_memory_usage_kb();
    auto time_start = std::chrono::high_resolution_clock::now();

    // --- 2. Initialization ---
    // Increased iterations for a more meaningful measurement
    const int ITERATIONS = 2000;
    auto n = Network(3, {4, 4, 1});

    std::vector<std::vector<double>> xs_raw = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };
    std::vector<double> ys_raw = {1.0, -1.0, 1.0, -1.0};


    for (int i = 0; i < ITERATIONS; ++i) {
        // --- Forward pass ---
        std::vector<std::shared_ptr<Value<double>>> y_pred;
        y_pred.reserve(xs_raw.size());
        for (const auto& x_row : xs_raw) {
            auto x_vals = convert_to_values(x_row);
            y_pred.push_back(n.forward(x_vals)[0]);
        }

        // --- Compute loss ---
        std::shared_ptr<Value<double>> loss = std::make_shared<Value<double>>(0.0);
        for (size_t j = 0; j < y_pred.size(); ++j) {
            auto y_true = std::make_shared<Value<double>>(ys_raw[j]);
            auto error = y_true - y_pred[j];
            loss = loss + (error ^ std::make_shared<Value<double>>(2.0));
        }

        // --- Zero gradients ---
        for (auto& p : n.parameters()) {
            p->grad = 0.0;
        }

        // --- Backward pass ---
        loss->backward();

        // --- Update parameters ---
        double learning_rate = 0.05;
        for (auto& p : n.parameters()) {
            p->data += -learning_rate * p->grad;
        }

        if (i % 200 == 0) { // Print progress periodically
             std::cout << "Iteration " << i << ", Loss: " << loss->data << std::endl;
        }
    }

    // --- 3. PROFILING REPORT ---
    auto time_end = std::chrono::high_resolution_clock::now();
    size_t mem_after = get_current_memory_usage_kb();

    std::chrono::duration<double, std::milli> time_elapsed_ms = time_end - time_start;
    size_t mem_used_kb = (mem_after > mem_before) ? (mem_after - mem_before) : 0;

    std::cout << "\n--- Profiling Results ---\n";
    std::cout << "Total execution time: " << time_elapsed_ms.count() << " ms\n";
    std::cout << "                       " << time_elapsed_ms.count() / 1000.0 << " s\n";
    std::cout << "-------------------------\n";
    std::cout << "Memory before loop: " << mem_before << " KB\n";
    std::cout << "Memory after loop:  " << mem_after << " KB\n";
    std::cout << "Approx. memory consumed by loop: " << mem_used_kb << " KB\n";
    std::cout << "-------------------------\n";


    return 0;
}
