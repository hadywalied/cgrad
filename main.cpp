#include <iostream>

#include "engine.h"
#include "neurals.h"
using namespace std;

int main() {
    // auto a = Value(1, "a");
    // auto b = Value(1, "b");
    // auto c = a + b;
    // c.label = "c";
    //
    // cout << "+" << endl;
    // cout << "c = " << c << endl;
    // cout << "c + 2 = " << c + 2 << endl;
    // cout << "1 + c= " << 1 + c << endl;
    //
    // cout << '-' << endl;
    // cout << "c - 2 = " << c - 2 << endl;
    // cout << "1 - c= " << 1 - c << endl;
    //
    // cout << '^' << endl;
    // cout << (c ^ a) << endl;
    // cout << (c ^ 2) << endl;
    // cout << (3 ^ c) << endl;
    //
    // cout << '/' << endl;
    // cout << (c / a) << endl;
    // cout << (c / 2) << endl;
    // cout << (2 / c) << endl;
    //
    // cout << '*' << endl;
    // cout << (c * a) << endl;
    // cout << (c * 2) << endl;
    // cout << (2 * c) << endl;
    //
    // cout << "tanh" << endl;
    // auto d = Value(0.5, "d");
    // cout << d.tanh() << endl;
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
    std::vector x = {Value(2.0), Value(3.0), Value(-1.0)};
    auto n = Network(3, {4, 4, 1});
    auto out = n.forward(x);
    for (auto &v: out) {
        cout << v << endl;
    }

    return 0;
}
