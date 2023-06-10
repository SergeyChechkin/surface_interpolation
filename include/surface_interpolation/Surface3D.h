/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <Eigen/Core>

namespace lib::surface_interpolation {

// 3D surface models

template<typename T>
class SurfacePolynomial_2 {
public:
    static constexpr size_t size_ = 6;
    using VectorT = Eigen::Vector<T, size_>;
public:
    // derivative of coefficients  
    static VectorT df_dc(T x, T y) {
        return {1, x, y, x * x, y * y, x * y};
    }

    // surface value
    static T val(T x, T y, const T cfs[size_]) {
        return df_dc(x,y).dot(VectorT(cfs));
    }
    
    // first partial derivatives
    static T df_dx(T x, T y, const T cfs[size_]) {
        return cfs[1] + 2 * cfs[3] * x + cfs[5] * y;
    }
    static T df_dy(T x, T y, const T cfs[size_]) {
        return cfs[2] + 2 * cfs[4] * y + cfs[5] * x;
    }
    
    // second partial derivatives
    static T df2_dxx(T x, T y, const T cfs[size_]) {
        return 2 * cfs[3];
    }
    static T df2_dyy(T x, T y, const T cfs[size_]) {
        return 2 * cfs[4];
    }
    static T df2_dxy(T x, T y, const T cfs[size_]) {
        return cfs[5];
    }
};

template<typename T>
class SurfacePolynomial_3 {
public:
    static constexpr size_t size_ = 10;
    using VectorT = Eigen::Vector<T, size_>;
public:
    // derivative of coefficients  
    static VectorT df_dc(T x, T y) {
        return {1, x, y, x * x, y * y, x * y, x * x * x, y * y * y, x * x * y, x * y * y};
    }

    // surface value
    static T val(T x, T y, const T cfs[size_]) {
        return df_dc(x,y).dot(VectorT(cfs));
    }
    
    // first partial derivatives
    static T df_dx(T x, T y, const T cfs[size_]) {
        return cfs[1] + 2 * cfs[3] * x + cfs[5] * y + 3 * cfs[6] * x * x + 2 * cfs[8] * x * y + cfs[9] * y * y;
    }
    static T df_dy(T x, T y, const T cfs[size_]) {
        return cfs[2] + 2 * cfs[4] * y + cfs[5] * x + 3 * cfs[7] * y * y + cfs[8] * x * x + 2 * cfs[9] * x * y;
    }
    
    // second partial derivatives
    static T df2_dxx(T x, T y, const T cfs[size_]) {
        return 2 * cfs[3] + 6 * cfs[6] * x + 2 * cfs[8] * y;
    }
    static T df2_dyy(T x, T y, const T cfs[size_]) {
        return 2 * cfs[4] + 6 * cfs[7] * y + 2 * cfs[9] * x;
    }
    static T df2_dxy(T x, T y, const T cfs[size_]) {
        return cfs[5] + 2 * cfs[8] * x + 2 * cfs[9] * y;
    }
};

}