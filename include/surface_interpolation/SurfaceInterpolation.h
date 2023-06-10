/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "surface_interpolation/Surface3D.h"
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <optional>

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
        //QuaternionT rotation_;
        //PointT origin_;
        CoefT coefficients_;
    };

    static std::optional<SurfacePatch> ComputeSurface(
        const PointsVectorT& patch_points,
        const PointT& center_point,
        T patch_radius) {

        if(auto plane_normal = PlaneNormal(patch_points, center_point)) {
            SurfacePatch result;
            // compute transfromation to local frame
            result.transform_.linear() = QuaternionT::FromTwoVectors(*plane_normal, PointT(0, 0, 1)).toRotationMatrix();
            result.transform_.translation() = -(result.transform_.linear() * center_point);
            // convert points to local plane frame
            PointsVectorT local_patch_points = result.transform_ * patch_points;

            return result;
        }

        return {};


/*
        /// convert points to local plane frame
        /// TODO: generate isometry transformation matrix
        std::vector<PointType> patch(points.Size());
        for(size_t i = 0; i < points.Size(); ++i) {
            patch[i] = result.rot_ * (points[i] - center_point);
        }

        DataMatType JtW;
        DataMatType J;
        DataMatType r;
        ComputeLSMats_<ModelType>(patch, JtW, J, r, patch_radius_sqr);

        /// compute coefficients
        auto coefficients = (JtW * J).inverse() * (JtW * r);
        /// TODO: find a mapping solution
        for(size_t i = 0; i < ModelType::size_; ++i)
            result.coefficients_[i] = coefficients(i, 0);
*/
    }
private:
    static std::optional<PointT> PlaneNormal(
        const PointsVectorT& patch_points,
        const PointT& center_point) {
        
        Eigen::MatrixX<T> centered = patch_points.rowwise() - center_point.transpose();
        Eigen::MatrixX<T> cov = (centered.adjoint() * centered) / T(patch_points.rows() - 1);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3<T>> eigen_solver(cov);

        if(Eigen::Success != eigen_solver.info())
            return {};

        return eigen_solver.eigenvectors().col(0);
    }

    static void ComputeMatsForLS_(
        const PointsVectorT& patch_points,
        Eigen::MatrixX<T>& JtW,
        Eigen::MatrixX<T>& J,
        Eigen::MatrixX<T>& r,
        T patch_radiuss) {
            
        }
};

}