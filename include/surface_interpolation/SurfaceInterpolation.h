/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "surface_interpolation/Surface3D.h"
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <optional>
#include <iostream>

namespace lib::surface_interpolation {

/// @brief Surface interpolation functions 
/// @tparam ModelType - surface model type 
/// @tparam T - scalar type (float, double) 
template <typename ModelType, typename T>
class SurfaceInterpolation {
public:
    using PointT = Eigen::Vector3<T>;
    using QuaternionT = Eigen::Quaternion<T>;
    using PointsVectorT = Eigen::Matrix<T, 3, Eigen::Dynamic>;
    using CoefT = typename ModelType::VectorT;
    using IsometryT = Eigen::Transform<T, 3, Eigen::Isometry>;
    
    struct SurfacePatch {
        IsometryT transform_;
        CoefT coefficients_;
    };

    /// @brief Compute surface patch, using patch mean point as a patch center.   
    /// @param patch_points - point cloud patch
    /// @param patch_radius - patch radius, only used for weight computation 
    /// @return - surface patch
    static std::optional<SurfacePatch> ComputeSurface(
        const PointsVectorT& patch_points,
        T patch_radius) {
            auto mean = patch_points.rowwise().mean();
            return ComputeSurface(patch_points, mean, patch_radius);
        }

    /// @brief Compute surface patch
    /// @param patch_points - point cloud patch
    /// @param center_point - patch center
    /// @param patch_radius - patch radius, only used for weight computation 
    /// @return - surface patch 
    static std::optional<SurfacePatch> ComputeSurface(
        const PointsVectorT& patch_points,
        const PointT& center_point,
        T patch_radius) {

        Eigen::MatrixX<T> centered = (patch_points.colwise() - center_point).transpose();
        Eigen::MatrixX<T> cov = (centered.adjoint() * centered) / T(patch_points.cols());

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3<T>> eigen_solver(cov);
        if(Eigen::Success != eigen_solver.info())
            return {};
            
        PointT plane_normal = eigen_solver.eigenvectors().col(0);

        SurfacePatch result;
        // compute transfromation to local frame
        result.transform_.linear() = QuaternionT::FromTwoVectors(plane_normal, PointT(0, 0, 1)).toRotationMatrix();
        result.transform_.translation() = -(result.transform_.linear() * center_point);
        
        // convert points to local plane frame
        PointsVectorT local_patch_points = result.transform_ * patch_points;
        // Interpolate surface with weighted least square 
        const T patch_radius_sqr = patch_radius * patch_radius;
        size_t size = local_patch_points.cols();
        Eigen::MatrixX<T> J = Eigen::MatrixX<T>(size, ModelType::size_);
        Eigen::MatrixX<T> r = Eigen::MatrixX<T>(size, 1);
        Eigen::MatrixX<T> W(size, 1);
        for(size_t i = 0; i < size; ++i) {
            const auto point = local_patch_points.col(i);
            J.row(i) = ModelType::df_dc(point[0], point[1]);
            r(i, 0) = point[2];
            W(i, 0) = std::exp(-point.squaredNorm() / patch_radius_sqr);
        }
        Eigen::MatrixX<T> JtW = J.transpose() * W.array().matrix().asDiagonal(); 
        result.coefficients_ = (JtW * J).inverse() * (JtW * r);

        return result;
    }

    /// @brief Compute surface patch
    /// @param patch_points - point cloud patch
    /// @return - surface patch 
    static std::optional<SurfacePatch> ComputeSurface(const PointsVectorT& patch_points) {
        auto mean = patch_points.rowwise().mean();
        Eigen::MatrixX<T> centered = (patch_points.colwise() - mean).transpose();
        Eigen::MatrixX<T> cov = (centered.adjoint() * centered) / T(patch_points.cols());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3<T>> eigen_solver(cov);

        if(Eigen::Success != eigen_solver.info())
            return {};
        
        SurfacePatch result;
        result.transform_.linear() = QuaternionT::FromTwoVectors(eigen_solver.eigenvectors().col(0), PointT(0, 0, 1)).toRotationMatrix();
        result.transform_.translation() = -(result.transform_.linear() * mean);
        auto eigenvalues = eigen_solver.eigenvalues(); 
        T patch_size = std::sqrt(eigenvalues[1]) + std::sqrt(eigenvalues[2]);
        // convert points to local plane frame
        PointsVectorT local_patch_points = result.transform_ * patch_points;
        // Interpolate surface with weigthed least sdquare 
        const T patch_radius_sqr = patch_size * patch_size;
        size_t size = local_patch_points.cols();
        Eigen::MatrixX<T> J = Eigen::MatrixX<T>(size, ModelType::size_);
        Eigen::MatrixX<T> r = Eigen::MatrixX<T>(size, 1);
        Eigen::MatrixX<T> W(size, 1);
        for(size_t i = 0; i < size; ++i) {
            const auto point = local_patch_points.col(i);
            J.row(i) = ModelType::df_dc(point[0], point[1]);
            r(i, 0) = point[2];
            W(i, 0) = std::exp(-point.squaredNorm() / patch_radius_sqr);
        }
        Eigen::MatrixX<T> JtW = J.transpose() * W.array().matrix().asDiagonal(); 
        result.coefficients_ = (JtW * J).inverse() * (JtW * r);

        return result;
    }

    /// @brief Compute shape operator for specified point, projected on surface. 
    /// @param surface - surface patch  
    /// @param point - point of interest 
    /// @return 2x2 shape operator matrix 
    static Eigen::Matrix2<T> ShapeOperator(const SurfacePatch& surface, const Eigen::Vector3<T>& point) {
        Eigen::Vector3<T> patch_points = surface.transform_ * point;
        T x = patch_points[0];
        T y = patch_points[1];
        // Surface derivatives
        T df_dx = ModelType::df_dx(x, y, surface.coefficients_.data());
        T df_dy = ModelType::df_dy(x, y, surface.coefficients_.data());
        T df2_dxx = ModelType::df2_dxx(x, y, surface.coefficients_.data());
        T df2_dyy = ModelType::df2_dyy(x, y, surface.coefficients_.data());
        T df2_dxy = ModelType::df2_dxy(x, y, surface.coefficients_.data());
        // Normal vector
        PointT n = PointT(-df_dx, -df_dy, 1).normalized();
        // Parametric surface
        PointT rx(1, 0, df_dx);
        PointT ry(0, 1, df_dy);
        PointT rxx(0, 0, df2_dxx);
        PointT ryy(0, 0, df2_dyy);
        PointT rxy(0, 0, df2_dxy);
        // First fundamental form
        T E = rx.dot(rx);
        T F = rx.dot(ry);
        T G = ry.dot(ry);
        // Second fundamental form
        T L = rxx.dot(n);
        T M = rxy.dot(n);
        T N = ryy.dot(n);
        Eigen::Matrix2<T> P1 {{L, M}, {M, N}};
        Eigen::Matrix2<T> P2 {{E, F}, {F, G}};
        return P1 * P2.inverse();
    }

    /// @brief Compute principal curvatures from shape operator 
    /// @param surface - surface patch  
    /// @param point - point of interest 
    /// @return - principal curvatures
    static std::optional<Eigen::Vector2<T>> PrincipalCurvatures(const SurfacePatch& surface, const Eigen::Vector3<T>& point) {
        Eigen::Matrix2<T> so = ShapeOperator(surface, point);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2<T>> eigen_solver(so);

        if(Eigen::Success != eigen_solver.info())
            return {};

        return eigen_solver.eigenvalues();
    }
};

}