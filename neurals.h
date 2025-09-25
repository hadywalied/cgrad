//
// Created by hady on 9/24/2025.
//

#ifndef NEURALS_H
#define NEURALS_H
#include <chrono>
#include <vector>

#include "engine.h"

#include <random>

class Neuron {
    Value<double> bias = Value(0.0);
    std::vector<Value<double> > weights = {};

public:
    Neuron(const int n_i) {
        constexpr double min_val = -1.0;
        constexpr double max_val = 1.0;
        const unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 generator(seed);

        std::uniform_real_distribution distribution(min_val, max_val);

        for (auto i = 0; i < n_i; ++i) {
            weights.emplace_back(Value(distribution(generator)));
        }
        bias = Value(distribution(generator));
    }

    Value<double> forward(const std::vector<Value<double> > &inputs) const {
        auto sum = bias;
        for (auto i = 0; i < inputs.size(); ++i) {
            sum = sum + inputs[i] * weights[i];
        }
        return sum.tanh();
    }

    std::vector<Value<double> > parameters() {
        auto all_params = weights;
        all_params.push_back(bias);
        return all_params;
    }
};

class Layer {
    std::vector<Neuron> neurons = {};

public:
    Layer(const int n_in, const int n_out) {
        for (auto i = 0; i < n_out; ++i) {
            neurons.emplace_back(n_in);
        }
    }

    std::vector<Value<double> > forward(const std::vector<Value<double> > &inputs) const {
        auto out = std::vector<Value<double> >();
        for (auto &neuron: neurons) {
            out.push_back(neuron.forward(inputs));
        }
        return out;
    }

    std::vector<Value<double> > parameters() {
        auto all_params = std::vector<Value<double> >();
        for (auto &neuron: neurons) {
            all_params.insert(all_params.end(), neuron.parameters().begin(), neuron.parameters().end());
        }
        return all_params;
    }
};

class Network {
    std::vector<Layer> layers = {};

public:
    Network(const int n_in, const std::vector<int> n_outs) {
        auto in_ = n_in;
        for (auto n_out: n_outs) {
            auto layer = Layer(in_, n_out);
            layers.push_back(layer);
            in_ = n_out;
        }
    }

    std::vector<Value<double> > forward(const std::vector<Value<double> > &inputs) const {
        auto out = inputs;
        for (auto &layer: layers) {
            out = layer.forward(out);
        }
        return out;
    }

    std::vector<Value<double> > parameters() {
        auto all_params = std::vector<Value<double> >();
        for (auto &layer: layers) {
            all_params.insert(all_params.end(), layer.parameters().begin(), layer.parameters().end());
        }
        return all_params;
    }
};


#endif //NEURALS_H
