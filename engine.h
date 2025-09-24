//
// Created by hady on 9/24/2025.
//

#ifndef ENGINE_H
#define ENGINE_H
#include <ostream>
#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include <set>
#include <functional>

template<typename T>
class Value {
public:
    T data;
    mutable double grad;
    std::string label;
    std::string operation;
    std::vector<std::reference_wrapper<const Value<T>> > children;
    std::function<void(Value<T> &)> _backward = [](Value<T> &) {
    };

    explicit Value(T data, std::string label = "",
                   std::string operation = "",
                   std::vector<std::reference_wrapper<const Value<T>> > children = {}) : data(data), grad(0.0),
        label(std::move(label)),
        operation(std::move(operation)),
        children(std::move(children)) {
    };

    template<typename U>
    friend std::ostream &operator<<(std::ostream &os, const Value<U> &value);

    Value<T> operator+(const Value<T> &other) const {
        auto out = Value<T>(data + other.data, "", "+", {*this, other});
        out._backward = [](Value<T> &value) {
            value.children[0].get().grad += value.grad;
            value.children[1].get().grad += value.grad;
        };
        return out;
    }

    Value<T> operator+(const T &other) const {
        return *this + Value<T>(other);
    }

    friend Value<T> operator+(const T &scalar, const Value<T> &value) {
        return value + scalar;
    }

    Value<T> operator-(const Value<T> &other) const {
        auto out = Value<T>(data - other.data, "", "-", {*this, other});
        out._backward = [](Value<T> &v) {
            v.children[0].get().grad += v.grad;
            v.children[1].get().grad -= v.grad;
        };
        return out;
    }

    Value<T> operator-(const T &other) const {
        return *this - Value<T>(other);
    }

    friend Value<T> operator-(const T &scalar, const Value<T> &value) {
        return Value<T>(scalar) - value;
    }

    Value<T> operator*(const Value<T> &other) const {
        auto out = Value<T>(data * other.data, "", "*", {*this, other});
        out._backward = [](Value<T> &v) {
            v.children[0].get().grad += v.children[1].get().data * v.grad;
            v.children[1].get().grad += v.children[0].get().data * v.grad;
        };
        return out;
    }

    Value<T> operator*(const T &other) const {
        return *this * Value<T>(other);
    }

    friend Value<T> operator*(const T &scalar, const Value<T> &value) {
        return value * scalar;
    }

    Value<T> operator^(const Value<T> &other) const {
        auto out = Value<T>(std::pow(data, other.data), "", "^", {*this, other});
        out._backward = [](Value<T> &v) {
            v.children[0].get().grad += (v.children[1].get().data * std::pow(
                                             v.children[0].get().data, v.children[1].get().data - 1)) * v.grad;
            v.children[1].get().grad += (std::log(v.children[0].get().data) * std::pow(
                                             v.children[0].get().data, v.children[1].get().data)) * v.grad;
        };
        return out;
    }

    Value<T> operator^(const T &other) const {
        return *this ^ Value<T>(other);
    }

    friend Value<T> operator^(const T &scalar, const Value<T> &value) {
        return Value<T>(scalar) ^ value;
    }

    Value<T> operator/(const Value<T> &other) const {
        return *this * (other ^ Value<T>(-1));
    }

    Value<T> operator/(const T &other) const {
        return *this / Value<T>(other);
    }

    friend Value<T> operator/(const T &scalar, const Value<T> &value) {
        return Value<T>(scalar) / value;
    }

    Value<double> tanh() const {
        auto vl = std::tanh(data);
        auto out = Value<double>(vl, "tanh(" + this->label + ")", "tanh", {*this});
        out._backward = [vl](Value<double> &v) {
            v.children[0].get().grad += (1 - std::pow(vl, 2)) * v.grad;
        };
        return out;
    };

    void backward() {
        std::vector<std::reference_wrapper<const Value<T>> > topo;
        std::set<const Value<T> *> visited;

        std::function<void(const Value<T> &)> build_topo = [&](const Value<T> &v) {
            if (visited.find(&v) == visited.end()) {
                visited.insert(&v);
                for (const auto &child: v.children) {
                    build_topo(child.get());
                }
                topo.push_back(v);
            }
        };

        build_topo(*this);

        grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            const_cast<Value<T> &>(it->get())._backward(const_cast<Value<T> &>(it->get()));
        }
    }
};

template<typename T>
std::ostream &operator<<(std::ostream &os, const Value<T> &value) {
    os << "Value(label= " << value.label << ", data=" << value.data << ", grad=" << value.grad << ", op= " << value.
            operation << ")";
    return os;
}

#endif //ENGINE_H
