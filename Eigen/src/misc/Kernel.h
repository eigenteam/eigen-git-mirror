// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_MISC_KERNEL_H
#define EIGEN_MISC_KERNEL_H

/** \class ei_kernel_return_value
  *
  */
template<typename DecompositionType>
struct ei_traits<ei_kernel_return_value<DecompositionType> >
{
  typedef typename DecompositionType::MatrixType MatrixType;
  typedef Matrix<
    typename MatrixType::Scalar,
    MatrixType::ColsAtCompileTime, // the number of rows in the "kernel matrix"
                                   // is the number of cols of the original matrix
                                   // so that the product "matrix * kernel = zero" makes sense
    Dynamic,                       // we don't know at compile-time the dimension of the kernel
    MatrixType::Options,
    MatrixType::MaxColsAtCompileTime, // see explanation for 2nd template parameter
    MatrixType::MaxColsAtCompileTime // the kernel is a subspace of the domain space,
                                     // whose dimension is the number of columns of the original matrix
  > ReturnMatrixType;
};

template<typename _DecompositionType> struct ei_kernel_return_value
 : public ReturnByValue<ei_kernel_return_value<_DecompositionType> >
{
  typedef _DecompositionType DecompositionType;
  const DecompositionType& m_dec;
  int m_rank, m_cols;

  ei_kernel_return_value(const DecompositionType& dec)
    : m_dec(dec),
      m_rank(dec.rank()),
      m_cols(m_rank==dec.cols() ? 1 : dec.cols() - m_rank)
  {}

  inline int rows() const { return m_dec.cols(); }
  inline int cols() const { return m_cols; }

  template<typename Dest> inline void evalTo(Dest& dst) const
  {
    static_cast<const ei_kernel_impl<DecompositionType, Dest> *>
      (this)->evalTo(dst);
  }
};

#endif // EIGEN_MISC_KERNEL_H
