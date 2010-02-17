// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2007-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_DIAGONALPRODUCT_H
#define EIGEN_DIAGONALPRODUCT_H

template<typename MatrixType, typename DiagonalType, int ProductOrder>
struct ei_traits<DiagonalProduct<MatrixType, DiagonalType, ProductOrder> >
 : ei_traits<MatrixType>
{
  typedef typename ei_scalar_product_traits<typename MatrixType::Scalar, typename DiagonalType::Scalar>::ReturnType Scalar;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    Flags = (HereditaryBits & (unsigned int)(MatrixType::Flags))
          | (PacketAccessBit & (unsigned int)(MatrixType::Flags) & (unsigned int)(DiagonalType::DiagonalVectorType::Flags)),
    CoeffReadCost = NumTraits<Scalar>::MulCost + MatrixType::CoeffReadCost + DiagonalType::DiagonalVectorType::CoeffReadCost
  };
};

template<typename MatrixType, typename DiagonalType, int ProductOrder>
class DiagonalProduct : ei_no_assignment_operator,
                        public MatrixBase<DiagonalProduct<MatrixType, DiagonalType, ProductOrder> >
{
  public:

    typedef MatrixBase<DiagonalProduct> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(DiagonalProduct)

    inline DiagonalProduct(const MatrixType& matrix, const DiagonalType& diagonal)
      : m_matrix(matrix), m_diagonal(diagonal)
    {
      ei_assert(diagonal.diagonal().size() == (ProductOrder == OnTheLeft ? matrix.rows() : matrix.cols()));
    }

    inline int rows() const { return m_matrix.rows(); }
    inline int cols() const { return m_matrix.cols(); }

    const Scalar coeff(int row, int col) const
    {
      return m_diagonal.diagonal().coeff(ProductOrder == OnTheLeft ? row : col) * m_matrix.coeff(row, col);
    }

    template<int LoadMode>
    EIGEN_STRONG_INLINE PacketScalar packet(int row, int col) const
    {
      enum {
        StorageOrder = Flags & RowMajorBit ? RowMajor : ColMajor,
        InnerSize = (MatrixType::Flags & RowMajorBit) ? MatrixType::ColsAtCompileTime : MatrixType::RowsAtCompileTime,
        DiagonalVectorPacketLoadMode = (LoadMode == Aligned && ((InnerSize%16) == 0)) ? Aligned : Unaligned
      };
      const int indexInDiagonalVector = ProductOrder == OnTheLeft ? row : col;

      if((int(StorageOrder) == RowMajor && int(ProductOrder) == OnTheLeft)
       ||(int(StorageOrder) == ColMajor && int(ProductOrder) == OnTheRight))
      {
        return ei_pmul(m_matrix.template packet<LoadMode>(row, col),
                       ei_pset1(m_diagonal.diagonal().coeff(indexInDiagonalVector)));
      }
      else
      {
        return ei_pmul(m_matrix.template packet<LoadMode>(row, col),
                       m_diagonal.diagonal().template packet<DiagonalVectorPacketLoadMode>(indexInDiagonalVector));
      }
    }

  protected:
    const typename MatrixType::Nested m_matrix;
    const typename DiagonalType::Nested m_diagonal;
};

/** \returns the diagonal matrix product of \c *this by the diagonal matrix \a diagonal.
  */
template<typename Derived>
template<typename DiagonalDerived>
inline const DiagonalProduct<Derived, DiagonalDerived, OnTheRight>
MatrixBase<Derived>::operator*(const DiagonalBase<DiagonalDerived> &diagonal) const
{
  return DiagonalProduct<Derived, DiagonalDerived, OnTheRight>(derived(), diagonal.derived());
}

/** \returns the diagonal matrix product of \c *this by the matrix \a matrix.
  */
template<typename DiagonalDerived>
template<typename MatrixDerived>
inline const DiagonalProduct<MatrixDerived, DiagonalDerived, OnTheLeft>
DiagonalBase<DiagonalDerived>::operator*(const MatrixBase<MatrixDerived> &matrix) const
{
  return DiagonalProduct<MatrixDerived, DiagonalDerived, OnTheLeft>(matrix.derived(), derived());
}


#endif // EIGEN_DIAGONALPRODUCT_H
