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

    static std::optional<SurfacePatch> ComputeSurface(
        const PointsVectorT& patch_points,
        T patch_radius) {
            auto mean = patch_points.rowwise().mean();
            return ComputeSurface(patch_points, mean, patch_radius);
        }

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
        // Interpolate surface with weigthed least sdquare 
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
        T patch_size = T(0.5) * (eigenvalues[1] + eigenvalues[2]);
        
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
};

}