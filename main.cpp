#include <iostream>

#include "engine.h"
using namespace std;

int main() {
    const auto a = Value(1);
    const auto b = Value(1);
    const auto c = a + b;

    cout <<"+"<< endl;
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

    return 0;
}
