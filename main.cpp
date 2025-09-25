#include <iostream>
#include <type_traits>
#include <memory>

#include "engine.h"
#include "neurals.h"
using namespace std;


template<typename T>
std::vector<std::shared_ptr<Value<double> > > convert_to_values(const std::vector<T> &input) {
    std::vector<std::shared_ptr<Value<double> > > result;
    result.reserve(input.size());

    if constexpr (std::is_arithmetic_v<T>) {
        for (const auto &val: input) {
            result.push_back(std::make_shared<Value<double> >(static_cast<double>(val)));
        }
    } else {
        for (const auto &val: input) {
            result.push_back(std::make_shared<Value<double> >(val));
        }
    }
    return result;
}

int main() {
    auto a = std::make_shared<Value<double>>(1, "a");
    auto b = std::make_shared<Value<double>>(1, "b");
    auto c = a + b;
    c.get()->label = "c";

    cout << "+" << endl;
    cout << "c = " << c << endl;
    cout << "c + 2 = " << c + 2 << endl;
    cout << "1 + c= " << 1 + c << endl;

    cout << '-' << endl;
    cout << "c - 2 = " << c - 2 << endl;
    cout << "1 - c= " << 1 - c << endl;

    cout << '^' << endl;
    cout << (c ^ a) << endl;
    cout << (c ^ 2) << endl;
    cout << (3 ^ c) << endl;

    cout << '/' << endl;
    cout << (c / a) << endl;
    cout << (c / 2) << endl;
    cout << (2 / c) << endl;

    cout << '*' << endl;
    cout << (c * a) << endl;
    cout << (c * 2) << endl;
    cout << (2 * c) << endl;

    cout << "tanh" << endl;
    auto d = std::make_shared<Value<double>>(0.5, "d");
    cout << tanh(d) << endl;
    //
    //
    // auto x = Value(2.0, "x");
    // auto y = Value(0.0, "y");
    // auto wx = Value(-3.0, "wx");
    // auto wy = Value(1.0, "wy");
    // auto bias = Value(6.88, "bias");
    // auto xwx = wx * x; xwx.label = "xwx";
    // auto ywy = wy * y; ywy.label = "ywy";
    // auto z = xwx + ywy; z.label = "x*wx + y*wy";
    // auto n = z + bias; n.label = 'n';
    // auto o = n.tanh(); o.label = 'o';
    // o.backward();
    //
    // cout << "x = " << x << endl;
    // cout << "y = " << y << endl;
    // cout << "wx = " << wx << endl;
    // cout << "wy = " << wy << endl;
    // cout << "bias = " << bias << endl;
    // cout << "xwx = " << xwx << endl;
    // cout << "ywy = " << ywy << endl;
    // cout << "z = " << z << endl;
    // cout << "n = " << n << endl;
    // cout << "o = " << o << endl;
    // cout << "o.grad = " << o.grad << endl;


    cout << "Trying Neurons" << endl;
    std::vector v = {Value(2.0), Value(3.0), Value(-1.0)};
    auto x = convert_to_values(v);
    auto n = Network(3, {4, 4, 1});
    auto out = n.forward(x);
    for (auto &v: out) {
        cout << v << endl;
    }


    // auto x = Value(2.0, "x");
    // auto y = Value(0.0, "y");
    // auto wx = Value(-3.0, "wx");
    // auto wy = Value(1.0, "wy");
    // auto bias = Value(6.88, "bias");
    // auto xwx = wx * x; xwx.label = "xwx";
    // auto ywy = wy * y; ywy.label = "ywy";
    // auto z = xwx + ywy; z.label = "x*wx + y*wy";
    // auto n = z + bias; n.label = 'n';
    // auto o = n.tanh(); o.label = 'o';
    // o.backward();
    //
    // cout << "x = " << x << endl;
    // cout << "y = " << y << endl;
    // cout << "wx = " << wx << endl;
    // cout << "wy = " << wy << endl;
    // cout << "bias = " << bias << endl;
    // cout << "xwx = " << xwx << endl;
    // cout << "ywy = " << ywy << endl;
    // cout << "z = " << z << endl;
    // cout << "n = " << n << endl;
    // cout << "o = " << o << endl;
    // cout << "o.grad = " << o.grad << endl;


    //
    // cout<<n.parameters().size()<<endl;
    //
    //
    cout << "making a training Loop" << endl;
    auto ITERATIONS = 20;

    std::vector<std::vector<Value<double> > > xs = {
    {Value(2.0), Value(3.0), Value(-1.0)},
    {Value(3.0), Value(-1.0), Value(-0.5)},
    {Value(0.5), Value(1.0), Value(1.0)},
    {Value(1.0), Value(1.0), Value(-1.0)}
    };
    std::vector ys = {Value(1.0), Value(-1.0), Value(1.0), Value(-1.0)}; // targets


    for (auto i = 0; i < ITERATIONS; ++i) {
    // Forward pass
    std::vector<std::vector<std::shared_ptr<Value<double>>>> y_pred = {};
    for (auto &x_: xs) {
    auto x_x = convert_to_values(x_);
    auto shared_ptrs = n.forward(x_x);
    y_pred.push_back(shared_ptrs);
    }

    // Zero gradients
    for (auto &p: n.parameters()) {
    p.get()->grad = 0.0;
    }

    // Compute loss
    auto y = std::make_shared<Value<double>>(ys[0]);
    auto yp = y_pred[0][0];
    auto loss = (y - yp) ^ std::make_shared<Value<double>>(2.0);
    for (auto j = 1; j < y_pred.size(); ++j) {
    y = std::make_shared<Value<double>>(ys[j]);
    yp = y_pred[j][0];
    loss = loss + ((y - yp) ^ std::make_shared<Value<double>>(2.0));
    }
    loss.get()->label = "loss";
    loss.get()->operation = "MSE";

    // Backward pass
    loss.get()->backward();

    // Update parameters
    double learning_rate = 0.05;
    for (auto &p: n.parameters()) {
    p.get()->data += -learning_rate * p.get()->grad;
    }

    cout << "i=" << i << " loss=" << loss.get()->data << endl;
    }
    return 0;
}
