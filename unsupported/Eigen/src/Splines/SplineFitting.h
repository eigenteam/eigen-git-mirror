// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 20010-2011 Hauke Heibel <hauke.heibel@gmail.com>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_SPLINE_FITTING_H
#define EIGEN_SPLINE_FITTING_H

#include <numeric>

#include "SplineFwd.h"

#include <Eigen/QR>

namespace Eigen
{
  template <typename KnotVectorType>
  void KnotAveraging(const KnotVectorType& parameters, DenseIndex degree, KnotVectorType& knots)
  {
    typedef typename KnotVectorType::Scalar Scalar;

    knots.resize(parameters.size()+degree+1);      

    for (DenseIndex j=1; j<parameters.size()-degree; ++j)
      knots(j+degree) = parameters.segment(j,degree).mean();

    knots.segment(0,degree+1) = KnotVectorType::Zero(degree+1);
    knots.segment(knots.size()-degree-1,degree+1) = KnotVectorType::Ones(degree+1);
  }

  template <typename PointArrayType, typename KnotVectorType>
  void ChordLengths(const PointArrayType& pts, KnotVectorType& chord_lengths)
  {
    typedef typename KnotVectorType::Scalar Scalar;

    const DenseIndex n = pts.cols();

    // 1. compute the column-wise norms
    chord_lengths.resize(pts.cols());
    chord_lengths[0] = 0;
    chord_lengths.rightCols(n-1) = (pts.array().leftCols(n-1) - pts.array().rightCols(n-1)).matrix().colwise().norm();

    // 2. compute the partial sums
    std::partial_sum(chord_lengths.data(), chord_lengths.data()+n, chord_lengths.data());

    // 3. normalize the data
    chord_lengths /= chord_lengths(n-1);
    chord_lengths(n-1) = Scalar(1);
  }

  template <typename SplineType>
  struct SplineFitting
  {
    template <typename PointArrayType>
    static SplineType Interpolate(const PointArrayType& pts, DenseIndex degree);
  };

  template <typename SplineType>
  template <typename PointArrayType>
  SplineType SplineFitting<SplineType>::Interpolate(const PointArrayType& pts, DenseIndex degree)
  {
    typedef typename SplineType::KnotVectorType::Scalar Scalar;      
    typedef typename SplineType::KnotVectorType KnotVectorType;
    typedef typename SplineType::BasisVectorType BasisVectorType;
    typedef typename SplineType::ControlPointVectorType ControlPointVectorType;      

    typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

    KnotVectorType chord_lengths; // knot parameters
    ChordLengths(pts, chord_lengths);

    KnotVectorType knots;
    KnotAveraging(chord_lengths, degree, knots);

    DenseIndex n = pts.cols();
    MatrixType A = MatrixType::Zero(n,n);
    for (DenseIndex i=1; i<n-1; ++i)
    {
      const DenseIndex span = SplineType::Span(chord_lengths[i], degree, knots);

      // The segment call should somehow be told the spline order at compile time.
      A.row(i).segment(span-degree, degree+1) = SplineType::BasisFunctions(chord_lengths[i], degree, knots);
    }
    A(0,0) = 1.0;
    A(n-1,n-1) = 1.0;

    HouseholderQR<MatrixType> qr(A);

    // Here, we are creating a temporary due to an Eigen issue.
    ControlPointVectorType ctrls = qr.solve(MatrixType(pts.transpose())).transpose();

    return SplineType(knots, ctrls);
  }
}

#endif // EIGEN_SPLINE_FITTING_H
