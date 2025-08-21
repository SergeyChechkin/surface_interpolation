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
    using MatrixT = Eigen::Matrix3<T>;
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
            const auto mean = patch_points.rowwise().mean();
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
        const auto cov = computeCovariance(patch_points, center_point);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3<T>> eigen_solver(cov);
        if(Eigen::Success != eigen_solver.info())
            return {};
            
        const PointT plane_normal = eigen_solver.eigenvectors().col(0);

        SurfacePatch result;
        // Compute transformation from point cloud frame to surface frame.
        result.transform_.linear() = QuaternionT::FromTwoVectors(plane_normal, PointT(0, 0, 1)).toRotationMatrix();
        result.transform_.translation() = -(result.transform_.linear() * center_point);
        
        // Interpolate surface with weighted least square 
        interpolateSurface(result, patch_points, patch_radius);
        return result;
    }

    /// @brief Compute surface patch. Extract additional parameters from the point cloud patch.
    /// @param patch_points - point cloud patch
    /// @return - surface patch 
    static std::optional<SurfacePatch> ComputeSurface(const PointsVectorT& patch_points) {
        const auto mean = patch_points.rowwise().mean();
        const auto cov = computeCovariance(patch_points, mean);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3<T>> eigen_solver(cov);
        if(Eigen::Success != eigen_solver.info())
            return {};
        
        SurfacePatch result;
        // Compute transformation from point cloud frame to surface frame.
        result.transform_.linear() = QuaternionT::FromTwoVectors(eigen_solver.eigenvectors().col(0), PointT(0, 0, 1)).toRotationMatrix();
        result.transform_.translation() = -(result.transform_.linear() * mean);
        const auto eigenvalues = eigen_solver.eigenvalues(); 
        T patch_size = std::sqrt(eigenvalues[1]) + std::sqrt(eigenvalues[2]);
        
        // Interpolate surface with weigthed least sdquare 
        interpolateSurface(result, patch_points, patch_size);
        return result;
    }

    /// @brief Compute shape operator for specified point, projected on surface. 
    /// @param surface - surface patch  
    /// @param point - point of interest 
    /// @return 2x2 shape operator matrix 
    static Eigen::Matrix2<T> ShapeOperator(const SurfacePatch& surface, const Eigen::Vector3<T>& point) {
        const Eigen::Vector3<T> patch_points = surface.transform_ * point;
        const T x = patch_points[0];
        const T y = patch_points[1];
        // Surface derivatives
        const T df_dx = ModelType::df_dx(x, y, surface.coefficients_.data());
        const T df_dy = ModelType::df_dy(x, y, surface.coefficients_.data());
        const T df2_dxx = ModelType::df2_dxx(x, y, surface.coefficients_.data());
        const T df2_dyy = ModelType::df2_dyy(x, y, surface.coefficients_.data());
        const T df2_dxy = ModelType::df2_dxy(x, y, surface.coefficients_.data());
        // Normal vector
        const PointT n = PointT(-df_dx, -df_dy, 1).normalized();
        // Parametric surface
        const PointT rx(1, 0, df_dx);
        const PointT ry(0, 1, df_dy);
        const PointT rxx(0, 0, df2_dxx);
        const PointT ryy(0, 0, df2_dyy);
        const PointT rxy(0, 0, df2_dxy);
        // First fundamental form
        const T E = rx.dot(rx);
        const T F = rx.dot(ry);
        const T G = ry.dot(ry);
        // Second fundamental form
        const T L = rxx.dot(n);
        const T M = rxy.dot(n);
        const T N = ryy.dot(n);
        const Eigen::Matrix2<T> P1 {{L, M}, {M, N}};
        const Eigen::Matrix2<T> P2 {{E, F}, {F, G}};
        return P1 * P2.inverse();
    }

    /// @brief Compute principal curvatures from shape operator 
    /// @param surface - surface patch  
    /// @param point - point of interest 
    /// @return - principal curvatures
    static std::optional<Eigen::Vector2<T>> PrincipalCurvatures(const SurfacePatch& surface, const Eigen::Vector3<T>& point) {
        const Eigen::Matrix2<T> so = ShapeOperator(surface, point);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2<T>> eigen_solver(so);

        if(Eigen::Success != eigen_solver.info())
            return {};

        return eigen_solver.eigenvalues();
    }
private:
    /// @brief Compute covariance matrix 
    /// @param patch_points - point cloud patch
    /// @param mean - patch center
    static MatrixT computeCovariance(const PointsVectorT& patch_points, const PointT& mean) {
        MatrixT result;
        result.setZero();

        const size_t size = patch_points.cols();
        for(size_t i = 0; i < size; ++i) {
            const PointT point = patch_points.col(i);
            const PointT centered_point = point - mean;
            result += centered_point * centered_point.transpose(); 
        }
        
        result *= 1.0 / T(size);

        return result;
    }
    
    /// @brief Interpolate surface with weigthed least sdquare  
    /// @param surface - surfase object
    /// @param patch_points - point cloud patch
    /// @param patch_size - patch radius
    static void interpolateSurface(
        SurfacePatch& surface, 
        const PointsVectorT& patch_points, 
        T patch_size) {
            const size_t size = patch_points.cols();  
            const T patch_radius_sqr = patch_size * patch_size;
            // TODO: pre-allocate this memory
            Eigen::MatrixX<T> J = Eigen::MatrixX<T>(size, ModelType::size_);
            Eigen::MatrixX<T> r = Eigen::MatrixX<T>(size, 1);
            Eigen::MatrixX<T> W(size, 1);
            for(size_t i = 0; i < size; ++i) {
                // convert point to local plane frame
                const auto point = surface.transform_ * patch_points.col(i);
                J.row(i) = ModelType::df_dc(point[0], point[1]);
                r(i, 0) = point[2];
                W(i, 0) = std::exp(-point.squaredNorm() / patch_radius_sqr);
            }
            const auto JtW = J.transpose() * W.array().matrix().asDiagonal(); 
            surface.coefficients_ = (JtW * J).inverse() * (JtW * r);
        }
};

}