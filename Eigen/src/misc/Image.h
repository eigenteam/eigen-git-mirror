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

#ifndef EIGEN_MISC_IMAGE_H
#define EIGEN_MISC_IMAGE_H

/** \class ei_image_return_value
  *
  */
template<typename DecompositionType>
struct ei_traits<ei_image_return_value<DecompositionType> >
{
  typedef typename DecompositionType::MatrixType MatrixType;
  typedef Matrix<
    typename MatrixType::Scalar,
    MatrixType::RowsAtCompileTime, // the image is a subspace of the destination space, whose
                                   // dimension is the number of rows of the original matrix
    Dynamic,                       // we don't know at compile time the dimension of the image (the rank)
    MatrixType::Options,
    MatrixType::MaxRowsAtCompileTime, // the image matrix will consist of columns from the original matrix,
    MatrixType::MaxColsAtCompileTime  // so it has the same number of rows and at most as many columns.
  > ReturnMatrixType;
};

template<typename _DecompositionType> struct ei_image_return_value
 : public ReturnByValue<ei_image_return_value<_DecompositionType> >
{
  typedef _DecompositionType DecompositionType;
  typedef typename DecompositionType::MatrixType MatrixType;

  const DecompositionType& m_dec;
  int m_rank, m_cols;
  const MatrixType& m_originalMatrix;

  ei_image_return_value(const DecompositionType& dec, const MatrixType& originalMatrix)
    : m_dec(dec), m_rank(dec.rank()),
      m_cols(m_rank == 0 ? 1 : m_rank),
      m_originalMatrix(originalMatrix)
  {}

  inline int rows() const { return m_dec.rows(); }
  inline int cols() const { return m_cols; }

  template<typename Dest> inline void evalTo(Dest& dst) const
  {
    static_cast<const ei_image_impl<DecompositionType>*>(this)->evalTo(dst);
  }
};

#define EIGEN_MAKE_IMAGE_HELPERS(DecompositionType) \
  typedef typename DecompositionType::MatrixType MatrixType; \
  typedef typename MatrixType::Scalar Scalar; \
  typedef typename MatrixType::RealScalar RealScalar; \
  inline const DecompositionType& dec() const { return this->m_dec; } \
  inline const MatrixType& originalMatrix() const { return this->m_originalMatrix; } \
  inline int rank() const { return this->m_rank; }

#endif // EIGEN_MISC_IMAGE_H
