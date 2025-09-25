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
    std::shared_ptr<Value<double>> bias = std::make_shared<Value<double>>(0.0);
    std::vector<std::shared_ptr<Value<double> >> weights = {};

public:
    Neuron(const int n_i) {
        constexpr double min_val = -1.0;
        constexpr double max_val = 1.0;
        const unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 generator(seed);

        std::uniform_real_distribution distribution(min_val, max_val);

        for (auto i = 0; i < n_i; ++i) {
            std::uniform_real_distribution<>::result_type rnd = distribution(generator);
            auto value = std::make_shared<Value<double>>(rnd);
            weights.emplace_back(value);
        }
        bias = std::make_shared<Value<double>>(distribution(generator));
    }

    std::shared_ptr<Value<double>> forward(const std::vector<std::shared_ptr<Value<double> >> &inputs) const {
        auto sum = bias;
        for (auto i = 0; i < inputs.size(); ++i) {
            sum = sum + (inputs[i] * weights[i]);
        }
        return tanh(sum);
    }

    std::vector<std::shared_ptr<Value<double> >> parameters() {
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

    std::vector<std::shared_ptr<Value<double> >> forward(const std::vector<std::shared_ptr<Value<double> >> &inputs) const {
        auto out = std::vector<std::shared_ptr<Value<double> >>();
        for (auto &neuron: neurons) {
            out.push_back(neuron.forward(inputs));
        }
        return out;
    }

    std::vector<std::shared_ptr<Value<double>>> parameters() {
        auto all_params = std::vector<std::shared_ptr<Value<double> >>();
        for (auto &neuron: neurons) {
            auto values = neuron.parameters();
            all_params.insert(all_params.end(), values.begin(), values.end());
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

    std::vector<std::shared_ptr<Value<double>> > forward(const std::vector<std::shared_ptr<Value<double>> > &inputs) const {
        auto out = inputs;
        for (auto &layer: layers) {
            out = layer.forward(out);
        }
        return out;
    }

    std::vector<std::shared_ptr<Value<double>>> parameters() {
        auto all_params = std::vector<std::shared_ptr<Value<double>>>();
        for (auto &layer: layers) {
            auto values = layer.parameters();
            all_params.insert(all_params.end(), values.begin(), values.end());
        }
        return all_params;
    }
};


#endif //NEURALS_H
