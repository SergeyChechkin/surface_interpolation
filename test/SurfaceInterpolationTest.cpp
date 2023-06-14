/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 


#include "surface_interpolation/Surface3D.h"
#include "surface_interpolation/SurfaceInterpolation.h"
#include <gtest/gtest.h>
#include <random>
#include <iostream>

using namespace lib::surface_interpolation;

template<typename T>
struct UnitSphereDistribution {
    Eigen::Vector3<T> operator()(std::default_random_engine& rng)
    {
        T angl = angl_dst_(rng);
        T z = z_dst_(rng);
        T scale = std::sqrt(1 - z * z);
        return {scale * std::cos(angl), scale * std::sin(angl), z};
    }

    std::uniform_real_distribution<T> angl_dst_ {0.0, 2 * M_PI};
    std::uniform_real_distribution<T> z_dst_ {-1, 1};
};

TEST(SurfaceInterpolation, SurfaceModel) { 
    SurfacePolynomial_2<float> model2;
    auto dfdc2 = model2.df_dc(1, 2);
    ASSERT_EQ(dfdc2[0], 1);
    ASSERT_EQ(dfdc2[1], 1);
    ASSERT_EQ(dfdc2[2], 2);
    ASSERT_EQ(dfdc2[3], 1);
    ASSERT_EQ(dfdc2[4], 4);
    ASSERT_EQ(dfdc2[5], 2);

    SurfacePolynomial_2<float>::VectorT coefs_2(1, 0, 0, 0, 0, 0);
    float val2 = model2.val(2, 2, coefs_2.data());
    ASSERT_EQ(val2, 1);

    SurfacePolynomial_3<float> model3;
    auto dfdc3 = model3.df_dc(1, 2);
    ASSERT_EQ(dfdc3[0], 1);
    ASSERT_EQ(dfdc3[1], 1);
    ASSERT_EQ(dfdc3[2], 2);
    ASSERT_EQ(dfdc3[3], 1);
    ASSERT_EQ(dfdc3[4], 4);
    ASSERT_EQ(dfdc3[5], 2);
    ASSERT_EQ(dfdc3[6], 1);
    ASSERT_EQ(dfdc3[7], 8);
    ASSERT_EQ(dfdc3[8], 2);
    ASSERT_EQ(dfdc3[9], 4);

    SurfacePolynomial_3<float>::VectorT coefs_3(1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    float val3 = model3.val(2, 2, coefs_3.data());
    ASSERT_EQ(val3, 1);
}

TEST(SurfaceInterpolation, SurfaceInterpolation_1) {
    SurfaceInterpolation<SurfacePolynomial_3<float>, float> algrtm;
    int points_count = 10000;
    float patch_size = 0.1;

    std::default_random_engine g;
    std::uniform_real_distribution<float> v(-patch_size, patch_size);
    Eigen::Matrix<float, 3, Eigen::Dynamic> points(3, points_count);
    for(int i = 0; i < points_count; ++i) {
        float x = v(g);
        float y = v(g);
        points.col(i) << x, y, 1.0f + 0.5 * x * x + 0.5 * y * y;
    }
    
    auto surf = algrtm.ComputeSurface(points, {0, 0, 1}, patch_size);
    
    double error = 0.001;
    ASSERT_NEAR(0, surf->coefficients_[0], error);
    ASSERT_NEAR(0, surf->coefficients_[1], error);
    ASSERT_NEAR(0, surf->coefficients_[2], error);
    ASSERT_NEAR(0.5, surf->coefficients_[3], error);
    ASSERT_NEAR(0.5, surf->coefficients_[4], error);
    ASSERT_NEAR(0, surf->coefficients_[5], error);
    ASSERT_NEAR(0, surf->coefficients_[6], error);
    ASSERT_NEAR(0, surf->coefficients_[7], error);
    ASSERT_NEAR(0, surf->coefficients_[8], error);
    ASSERT_NEAR(0, surf->coefficients_[9], error);

    auto surf_1 = algrtm.ComputeSurface(points);
    std::cout << surf_1->coefficients_.transpose() << std::endl;

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}