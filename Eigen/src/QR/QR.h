// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_QR_H
#define EIGEN_QR_H

/** \class QR
  *
  * \brief QR decomposition of a matrix
  *
  * \param MatrixType the type of the matrix of which we are computing the QR decomposition
  *
  * This class performs a QR decomposition using Householder transformations. The result is
  * stored in a compact way.
  *
  * \todo add convenient method to direclty use the result in a compact way. First need to determine
  * typical use cases though.
  *
  * \todo what about complex matrices ?
  *
  * \sa MatrixBase::qr()
  */
template<typename MatrixType> class QR
{
  public:

    typedef typename MatrixType::Scalar Scalar;
    typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, MatrixType::ColsAtCompileTime> MatrixTypeR;
    typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> VectorType;

    QR(const MatrixType& matrix)
      : m_qr(matrix.rows(), matrix.cols()),
        m_norms(matrix.cols())
    {
      _compute(matrix);
    }

    /** \returns whether or not the matrix is of full rank */
    bool isFullRank() const { return ei_isMuchSmallerThan(m_norms.cwiseAbs().minCoeff(), Scalar(1)); }

    MatrixTypeR matrixR(void) const;

    MatrixType matrixQ(void) const;

  private:

    void _compute(const MatrixType& matrix);

  protected:
    MatrixType m_qr;
    VectorType m_norms;
};

template<typename MatrixType>
void QR<MatrixType>::_compute(const MatrixType& matrix)
{
  m_qr = matrix;
  int rows = matrix.rows();
  int cols = matrix.cols();

  for (int k = 0; k < cols; k++)
  {
    int remainingSize = rows-k;

    Scalar nrm = m_qr.col(k).end(remainingSize).norm();

    if (nrm != Scalar(0))
    {
      // form k-th Householder vector
      if (m_qr(k,k) < 0)
        nrm = -nrm;

      m_qr.col(k).end(rows-k) /= nrm;
      m_qr(k,k) += 1.0;

      // apply transformation to remaining columns
      int remainingCols = cols - k -1;
      if (remainingCols>0)
      {
        m_qr.corner(BottomRight, remainingSize, remainingCols) -= (1./m_qr(k,k)) * m_qr.col(k).end(remainingSize)
          * (m_qr.col(k).end(remainingSize).transpose() * m_qr.corner(BottomRight, remainingSize, remainingCols));
      }
    }
    m_norms[k] = -nrm;
  }
}

/** \returns the matrix R */
template<typename MatrixType>
typename QR<MatrixType>::MatrixTypeR QR<MatrixType>::matrixR(void) const
{
  int cols = m_qr.cols();
  MatrixTypeR res = m_qr.block(0,0,cols,cols).template extract<StrictlyUpper>();
  res.diagonal() = m_norms;
  return res;
}

/** \returns the matrix Q */
template<typename MatrixType>
MatrixType QR<MatrixType>::matrixQ(void) const
{
  int rows = m_qr.rows();
  int cols = m_qr.cols();
  MatrixType res = MatrixType::identity(rows, cols);
  for (int k = cols-1; k >= 0; k--)
  {
    for (int j = k; j < cols; j++)
    {
      if (res(k,k) != Scalar(0))
      {
        int endLength = rows-k;
        Scalar s = -(m_qr.col(k).end(endLength).transpose() * res.col(j).end(endLength))(0,0) / m_qr(k,k);

        res.col(j).end(endLength) += s * m_qr.col(k).end(endLength);
      }
    }
  }
  return res;
}

/** \return the QR decomposition of \c *this.
  *
  * \sa class QR
  */
template<typename Derived>
const QR<typename ei_eval<Derived>::type>
MatrixBase<Derived>::qr() const
{
  return QR<typename ei_eval<Derived>::type>(derived());
}


#endif // EIGEN_QR_H
