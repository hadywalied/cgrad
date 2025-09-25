# cgrad

cgrad is a lightweight automatic differentiation engine written in C++. It implements a `Value` class that tracks operations to build a computation graph. This allows for automatic calculation of gradients using backpropagation. The library also includes a small neural network library (`neurals.h`) to build and train multi-layer perceptrons. This project is inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Features

*   Scalar `Value` class with automatic gradient calculation.
*   Supports basic arithmetic operations: `+`, `-`, `*`, `/`, `^` (power).
*   Activation function: `tanh`.
*   Build and train multi-layer perceptrons (MLPs).
*   Header-only library for easy integration.

## Getting Started

The project uses CMake to build.

```bash
git clone https://github.com/hadywalied/cgrad.git
cd cgrad
mkdir build
cd build
cmake ..
make
```

This will build the executable `cgrad` in the `build` directory.

## Usage

Here is an example of how to define a simple expression and compute the gradients:

```cpp
#include <iostream>
#include <memory>
#include "engine.h"

int main() {
    auto a = std::make_shared<Value<double>>(2.0, "a");
    auto b = std::make_shared<Value<double>>(-3.0, "b");
    auto c = std::make_shared<Value<double>>(10.0, "c");
    auto e = a * b; e->label = "e";
    auto d = e + c; d->label = "d";
    auto f = std::make_shared<Value<double>>(-2.0, "f");
    auto L = d * f; L->label = "L";

    L->backward();

    std::cout << "L=" << L->data << std::endl;
    std::cout << "d=" << d->data << ", d.grad=" << d->grad << std::endl;
    std::cout << "f=" << f->data << ", f.grad=" << f->grad << std::endl;
    std::cout << "e=" << e->data << ", e.grad=" << e->grad << std::endl;
    std::cout << "c=" << c->data << ", c.grad=" << c->grad << std::endl;
    std::cout << "a=" << a->data << ", a.grad=" << a->grad << std::endl;
    std::cout << "b=" << b->data << ", b.grad=" << b->grad << std::endl;

    return 0;
}
```

Here is an example of how to build and train a neural network:

```cpp
#include <iostream>
#include <vector>
#include <memory>
#include "neurals.h"

int main() {
    // Create a network with 3 inputs, two hidden layers of 4 neurons each, and 1 output
    auto n = Network(3, {4, 4, 1});

    // Training data
    std::vector<std::vector<double>> xs_data = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };
    std::vector<double> ys_data = {1.0, -1.0, -1.0, 1.0};

    // Convert data to Value objects
    std::vector<std::vector<std::shared_ptr<Value<double>>>> xs;
    for (const auto& x_row : xs_data) {
        std::vector<std::shared_ptr<Value<double>>> row;
        for (const auto& val : x_row) {
            row.push_back(std::make_shared<Value<double>>(val));
        }
        xs.push_back(row);
    }

    std::vector<std::shared_ptr<Value<double>>> ys;
    for (const auto& val : ys_data) {
        ys.push_back(std::make_shared<Value<double>>(val));
    }


    // Training loop
    for (int i = 0; i < 100; ++i) {
        // Forward pass
        std::vector<std::shared_ptr<Value<double>>> ypred;
        for (const auto& x : xs) {
            auto y = n.forward(x);
            ypred.push_back(y[0]);
        }

        // Compute loss (Mean Squared Error)
        auto loss = std::make_shared<Value<double>>(0.0);
        for (size_t j = 0; j < ys.size(); ++j) {
            auto diff = ypred[j] - ys[j];
            loss = loss + (diff * diff);
        }

        // Zero gradients
        for (auto& p : n.parameters()) {
            p->grad = 0.0;
        }

        // Backward pass
        loss->backward();

        // Update parameters
        double learning_rate = 0.05;
        for (auto& p : n.parameters()) {
            p->data -= learning_rate * p->grad;
        }

        std::cout << "Iteration " << i << ", Loss: " << loss->data << std::endl;
    }

    return 0;
}
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License.