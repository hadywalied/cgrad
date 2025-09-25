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
#include <memory> // Add this include for shared_ptr

template<typename T>
class Value {
public:
    T data;
    mutable double grad;
    std::string label;
    std::string operation;
    std::vector<std::shared_ptr<Value<T>>> children;
    std::function<void(Value<T> &)> _backward = [](Value<T> &) {};

    Value(T data, std::string label = "",
                   std::string operation = "",
                   std::vector<std::shared_ptr<Value<T>>> children_vector = {}) :
        data(data),
        grad(0.0),
        label(std::move(label)),
        operation(std::move(operation)) {
        children = std::move(children_vector);
    }

    void backward() {
        std::vector<Value<T> *> topo;
        std::set<Value<T> *> visited;

        std::function<void(Value<T> *)> build_topo = [&](Value<T> *v) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (const auto &child: v->children) {
                    build_topo(child.get());
                }
                topo.push_back(v);
            }
        };

        build_topo(this);

        grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward(**it);
        }
    }
};

template<typename T>
std::shared_ptr<Value<T>> operator+(const std::shared_ptr<Value<T>> &lhs, const std::shared_ptr<Value<T>> &rhs) {
    T args = lhs->data + rhs->data;
    std::vector<std::shared_ptr<Value<T>>> children = {lhs, rhs};
    auto out = std::make_shared<Value<T>>(args, "", "*", children);
    out->_backward = [](Value<T> &value) {
        value.children[0]->grad += value.grad;
        value.children[1]->grad += value.grad;
    };
    return out;
}

template<typename T>
std::shared_ptr<Value<T>> operator-(const std::shared_ptr<Value<T>> &lhs, const std::shared_ptr<Value<T>> &rhs) {
    auto args = lhs->data - rhs->data;
    std::vector<std::shared_ptr<Value<T>>> children = {lhs, rhs};
    auto out = std::make_shared<Value<T>>(args, "", "*", children);
    out->_backward = [](Value<T> &v) {
        v.children[0]->grad += v.grad;
        v.children[1]->grad -= v.grad;
    };
    return out;
}

template<typename T>
std::shared_ptr<Value<T>> operator*(const std::shared_ptr<Value<T>> &lhs, const std::shared_ptr<Value<T>> &rhs) {
    T args = (lhs->data * rhs->data);
    std::vector<std::shared_ptr<Value<T>>> children = {lhs, rhs};
    auto out = std::make_shared<Value<T>>(args, "", "*", children);
    out->_backward = [](Value<T> &v) {
        v.children[0]->grad += v.children[1]->data * v.grad;
        v.children[1]->grad += v.children[0]->data * v.grad;
    };
    return out;
}

template<typename T>
std::shared_ptr<Value<T>> operator^(const std::shared_ptr<Value<T>> &lhs, const std::shared_ptr<Value<T>> &rhs) {
    auto args = std::pow(lhs->data, rhs->data);
    std::vector<std::shared_ptr<Value<T>>> children = {lhs, rhs};
    auto out = std::make_shared<Value<T>>(args, "", "*", children);
    out->_backward = [](Value<T> &v) {
        v.children[0]->grad += (v.children[1]->data * std::pow(
                                             v.children[0]->data, v.children[1]->data - 1)) * v.grad;
        v.children[1]->grad += (std::log(v.children[0]->data) * std::pow(
                                             v.children[0]->data, v.children[1]->data)) * v.grad;
    };
    return out;
}

template<typename T>
std::shared_ptr<Value<T>> operator/(const std::shared_ptr<Value<T>> &lhs, const std::shared_ptr<Value<T>> &rhs) {
    return lhs * (rhs ^ std::make_shared<Value<T>>(-1));
}

template<typename T>
std::shared_ptr<Value<T>> operator+(const std::shared_ptr<Value<T>> &lhs, const T &rhs) {
    return lhs + std::make_shared<Value<T>>(rhs);
}

template<typename T>
std::shared_ptr<Value<T>> operator+(const T &lhs, const std::shared_ptr<Value<T>> &rhs) {
    return std::make_shared<Value<T>>(lhs) + rhs;
}

template<typename T>
std::shared_ptr<Value<T>> operator-(const std::shared_ptr<Value<T>> &lhs, const T &rhs) {
    return lhs - std::make_shared<Value<T>>(rhs);
}

template<typename T>
std::shared_ptr<Value<T>> operator-(const T &lhs, const std::shared_ptr<Value<T>> &rhs) {
    return std::make_shared<Value<T>>(lhs) - rhs;
}

template<typename T>
std::shared_ptr<Value<T>> operator*(const std::shared_ptr<Value<T>> &lhs, const T &rhs) {
    return lhs * std::make_shared<Value<T>>(rhs);
}

template<typename T>
std::shared_ptr<Value<T>> operator*(const T &lhs, const std::shared_ptr<Value<T>> &rhs) {
    return std::make_shared<Value<T>>(lhs) * rhs;
}

template<typename T>
std::shared_ptr<Value<T>> operator^(const std::shared_ptr<Value<T>> &lhs, const T &rhs) {
    return lhs ^ std::make_shared<Value<T>>(rhs);
}

template<typename T>
std::shared_ptr<Value<T>> operator^(const T &lhs, const std::shared_ptr<Value<T>> &rhs) {
    return std::make_shared<Value<T>>(lhs) ^ rhs;
}

template<typename T>
std::shared_ptr<Value<T>> operator/(const std::shared_ptr<Value<T>> &lhs, const T &rhs) {
    return lhs / std::make_shared<Value<T>>(rhs);
}

template<typename T>
std::shared_ptr<Value<T>> operator/(const T &lhs, const std::shared_ptr<Value<T>> &rhs) {
    return std::make_shared<Value<T>>(lhs) / rhs;
}

template<typename T>
std::shared_ptr<Value<T>> tanh(const std::shared_ptr<Value<T>> &val) {
    auto vl = std::tanh(val->data);
    std::vector children = {val, };
    auto out = std::make_shared<Value<T>>(vl, "", "tanh", children);
    out->_backward = [vl](Value<T> &v) {
        auto g = (1 - std::pow(vl, 2)) * v.grad;
        v.children[0]->grad += g;
    };
    return out;
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const Value<T> &value) {
    os << "Value(label= " << value.label << ", data=" << value.data << ", grad=" << value.grad << ", op= " << value.
            operation << ")";
    return os;
}

#endif //ENGINE_H
