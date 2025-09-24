//
// Created by hady on 9/24/2025.
//

#ifndef ENGINE_H
#define ENGINE_H
#include <ostream>


template<typename T>
class Value {
public:
    T data;
    float grad;

    explicit Value(T data) : data(data), grad(0) {
    };

    template<typename U>
    friend std::ostream &operator<<(std::ostream &os, const Value<U> &value);

    Value<T> operator+(const Value<T> &other) const {
        return Value<T>(data + other.data);
    }

    Value<T> operator+(const T &other) const {
        return Value<T>(data + other);
    }

    // scalar plus Value (needs to be a friend function)
    friend Value<T> operator+(const T &scalar, const Value<T> &value) {
        return Value<T>(scalar + value.data);
    }


    Value<T> operator-(const Value &other) const {
        return Value<T>(data - other.data);
    }

    Value<T> operator-(const T &other) const {
        return Value<T>(data - other);
    }

    friend Value<T> operator-(const T &scalar, const Value<T> &value) {
        return Value<T>(scalar - value.data);
    }

    Value<T> operator*(const Value &other) const {
        return Value<T>(data * other.data);
    }

    Value<T> operator*(const T &other) const {
        return Value<T>(data * other);
    }

    friend Value<T> operator*(const T &scalar, const Value<T> &value) {
        return Value<T>(scalar * value.data);
    }

    Value<T> operator^(const Value &other) const {
        if (other.data == 0) {
            return Value<T>(1);
        }
        if (other.data == 1) {
            return Value<T>(data);
        }
        T result = 1;
        T exponent = other.data;
        T value = data;
        // Handle negative exponents
        if (exponent < 0) {
            value = 1 / value;
            exponent = -exponent;
        }
        // Use binary exponentiation for better efficiency
        while (exponent > 0) {
            if (exponent & 1) {
                result *= value;
            }
            value *= value;
            exponent >>= 1;
        }
        return Value<T>(result);
    }

    Value<T> operator^(const T &other) const {
        if (other == 0) {
            return Value<T>(1);
        }
        if (other == 1) {
            return Value<T>(data);
        }
        T result = 1;
        T exponent = other;
        T value = data;
        // Handle negative exponents
        if (exponent < 0) {
            value = 1 / value;
            exponent = -exponent;
        }
        // Use binary exponentiation for better efficiency
        while (exponent > 0) {
            if (exponent & 1) {
                result *= value;
            }
            value *= value;
            exponent >>= 1;
        }
        return Value<T>(result);
    }

    friend Value<T> operator^(const T &scalar, const Value<T> &value_obj_value) {
        if (value_obj_value.data == 0) {
            return Value<T>(1);
        }
        if (value_obj_value.data == 1) {
            return Value<T>(scalar);
        }
        T result = 1;
        T exponent = value_obj_value.data;
        T value = scalar;
        // Handle negative exponents
        if (exponent < 0) {
            value = 1 / value;
            exponent = -exponent;
        }
        // Use binary exponentiation for better efficiency
        while (exponent > 0) {
            if (exponent & 1) {
                result *= value;
            }
            value *= value;
            exponent >>= 1;
        }
        return Value<T>(result);
    }

    Value<T> operator/(const Value &other) const {
        return (*this) * (other ^ (-1));
    }


    Value<T> operator/(const T &other) const {
        return Value((*this).data / other);
    }

    friend Value<T> operator/(const T &scalar, const Value<T> &value) {
        return Value(scalar / value.data);
    }

    //     Value<T> tanh();
    //
    //     void backward();
    //
    // private:
    //     void _backward();
};

template<typename T>
std::ostream &operator<<(std::ostream &os, const Value<T> &value) {
    os << "Value(data=" << value.data << ", grad=" << value.grad << ")";
    return os;
}

#endif //ENGINE_H
