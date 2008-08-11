// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_REGRESSION_H
#define EIGEN_REGRESSION_H

/** \ingroup Regression_Module
  *
  * \regression_module
  *
  * For a set of points, this function tries to express
  * one of the coords as a linear (affine) function of the other coords.
  *
  * This is best explained by an example. This function works in full
  * generality, for points in a space of arbitrary dimension, and also over
  * the complex numbers, but for this example we will work in dimension 3
  * over the real numbers (doubles).
  *
  * So let us work with the following set of 5 points given by their
  * \f$(x,y,z)\f$ coordinates:
  * @code
    Vector3d points[5];
    points[0] = Vector3d( 3.02, 6.89, -4.32 );
    points[1] = Vector3d( 2.01, 5.39, -3.79 );
    points[2] = Vector3d( 2.41, 6.01, -4.01 );
    points[3] = Vector3d( 2.09, 5.55, -3.86 );
    points[4] = Vector3d( 2.58, 6.32, -4.10 );
  * @endcode
  * Suppose that we want to express the second coordinate (\f$y\f$) as a linear
  * expression in \f$x\f$ and \f$z\f$, that is,
  * \f[ y=ax+bz+c \f]
  * for some constants \f$a,b,c\f$. Thus, we want to find the best possible
  * constants \f$a,b,c\f$ so that the plane of equation \f$y=ax+bz+c\f$ fits
  * best the five above points. To do that, call this function as follows:
  * @code
    Vector3d coeffs; // will store the coefficients a, b, c
    linearRegression(
      5,
      points,
      &coeffs,
      1 // the coord to express as a function of
        // the other ones. 0 means x, 1 means y, 2 means z.
    );
  * @endcode
  * Now the vector \a coeffs is approximately
  * \f$( 0.495 ,  -1.927 ,  -2.906 )\f$.
  * Thus, we get \f$a=0.495, b = -1.927, c = -2.906\f$. Let us check for
  * instance how near points[0] is from the plane of equation \f$y=ax+bz+c\f$.
  * Looking at the coords of points[0], we see that:
  * \f[ax+bz+c = 0.495 * 3.02 + (-1.927) * (-4.32) + (-2.906) = 6.91.\f]
  * On the other hand, we have \f$y=6.89\f$. We see that the values
  * \f$6.91\f$ and \f$6.89\f$
  * are near, so points[0] is very near the plane of equation \f$y=ax+bz+c\f$.
  *
  * Let's now describe precisely the parameters:
  * @param numPoints the number of points
  * @param points the array of pointers to the points on which to perform the linear regression
  * @param retCoefficients pointer to the vector in which to store the result.
                           This vector must be of the same type and size as the
                           data points. The meaning of its coords is as follows.
                           For brevity, let \f$n=Size\f$,
                           \f$r_i=retCoefficients[i]\f$,
                           and \f$f=funcOfOthers\f$. Denote by
                           \f$x_0,\ldots,x_{n-1}\f$
                           the n coordinates in the n-dimensional space.
                           Then the result equation is:
                           \f[ x_f = r_0 x_0 + \cdots + r_{f-1}x_{f-1}
                            + r_{f+1}x_{f+1} + \cdots + r_{n-1}x_{n-1} + r_n. \f]
  * @param funcOfOthers Determines which coord to express as a function of the
                        others. Coords are numbered starting from 0, so that a
                        value of 0 means \f$x\f$, 1 means \f$y\f$,
                        2 means \f$z\f$, ...
  *
  * \sa fitHyperplane()
  */
template<typename VectorType>
void linearRegression(int numPoints,
                      VectorType **points,
                      VectorType *result,
                      int funcOfOthers )
{
  typedef typename VectorType::Scalar Scalar;
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(VectorType)
  ei_assert(numPoints >= 1);
  int size = points[0]->size();
  ei_assert(funcOfOthers >= 0 && funcOfOthers < size);
  result->resize(size);

  Matrix<Scalar, Dynamic, VectorType::SizeAtCompileTime,
         Dynamic, VectorType::MaxSizeAtCompileTime, RowMajorBit>
    m(numPoints, size);
  if(funcOfOthers>0)
    for(int i = 0; i < numPoints; i++)
      m.row(i).start(funcOfOthers) = points[i]->start(funcOfOthers);
  if(funcOfOthers<size-1)
    for(int i = 0; i < numPoints; i++)
      m.row(i).block(funcOfOthers, size-funcOfOthers-1)
        = points[i]->end(size-funcOfOthers-1);
  for(int i = 0; i < numPoints; i++)
    m.row(i).coeffRef(size-1) = Scalar(1);

  VectorType v(size);
  v.setZero();
  for(int i = 0; i < numPoints; i++)
    v += m.row(i).adjoint() * points[i]->coeff(funcOfOthers);

  ei_assert((m.adjoint()*m).lu().solve(v, result));
}

/** \ingroup Regression_Module
  *
  * \regression_module
  *
  * This function is quite similar to linearRegression(), so we refer to the
  * documentation of this function and only list here the differences.
  *
  * The main difference from linearRegression() is that this function doesn't
  * take a \a funcOfOthers argument. Instead, it finds a general equation
  * of the form
  * \f[ r_0 x_0 + \cdots + r_{n-1}x_{n-1} + r_n = 0, \f]
  * where \f$n=Size\f$, \f$r_i=retCoefficients[i]\f$, and we denote by
  * \f$x_0,\ldots,x_{n-1}\f$ the n coordinates in the n-dimensional space.
  *
  * Thus, the vector \a retCoefficients has size \f$n+1\f$, which is another
  * difference from linearRegression().
  *
  * This functions proceeds by first determining which coord has the smallest variance,
  * and then calls linearRegression() to express that coord as a function of the other ones.
  *
  * \sa linearRegression()
  */
template<typename VectorType, typename BigVectorType>
void fitHyperplane(int numPoints,
                   VectorType **points,
                   BigVectorType *result)
{
  typedef typename VectorType::Scalar Scalar;
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(VectorType)
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(BigVectorType)
  ei_assert(numPoints >= 1);
  int size = points[0]->size();
  ei_assert(size+1 == result->size());

  // now let's find out which coord varies the least. This is
  // approximative. All that matters is that we don't pick a coordinate
  // that varies orders of magnitude more than another one.
  VectorType mean(size);
  Matrix<typename NumTraits<Scalar>::Real,
         VectorType::RowsAtCompileTime, VectorType::ColsAtCompileTime,
         VectorType::MaxRowsAtCompileTime, VectorType::MaxColsAtCompileTime
        > variance(size);
  mean.setZero();
  variance.setZero();
  for(int i = 0; i < numPoints; i++)
    mean += *(points[i]);
  mean /= numPoints;
  for(int j = 0; j < size; j++)
  {
    for(int i = 0; i < numPoints; i++)
      variance.coeffRef(j) += ei_abs2(points[i]->coeff(j) - mean.coeff(j));
  }

  int coord_min_variance;
  variance.minCoeff(&coord_min_variance);

  // let's now perform a linear regression with respect to that
  // not-too-much-varying coord
  VectorType affine(size);
  linearRegression(numPoints, points, &affine, coord_min_variance);

  if(coord_min_variance>0)
    result->start(coord_min_variance) = affine.start(coord_min_variance);
  result->coeffRef(coord_min_variance) = static_cast<Scalar>(-1);
  result->end(size-coord_min_variance) = affine.end(size-coord_min_variance);
}


#endif // EIGEN_REGRESSION_H
