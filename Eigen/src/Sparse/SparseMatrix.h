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

#ifndef EIGEN_SPARSEMATRIX_H
#define EIGEN_SPARSEMATRIX_H

template<typename _Scalar> class SparseMatrix;

/** \class SparseMatrix
  *
  * \brief Sparse matrix
  *
  * \param _Scalar the scalar type, i.e. the type of the coefficients
  *
  * See http://www.netlib.org/linalg/html_templates/node91.html for details on the storage scheme.
  *
  */
template<typename _Scalar>
struct ei_traits<SparseMatrix<_Scalar> >
{
  typedef _Scalar Scalar;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    Flags = 0,
    CoeffReadCost = NumTraits<Scalar>::ReadCost
  };
};

template<typename _Scalar>
class SparseMatrix : public MatrixBase<SparseMatrix<_Scalar> >
{
  public:
    EIGEN_GENERIC_PUBLIC_INTERFACE(SparseMatrix)

  protected:

    int* m_colPtrs;
    SparseArray<Scalar> m_data;
    int m_rows;
    int m_cols;

    inline int _rows() const { return m_rows; }
    inline int _cols() const { return m_cols; }

    inline const Scalar& _coeff(int row, int col) const
    {
      int id = m_colPtrs[col];
      int end = m_colPtrs[col+1];
      while (id<end && m_data.index(id)!=row)
      {
        ++id;
      }
      if (id==end)
        return 0;
      return m_data.value(id);
    }

    inline Scalar& _coeffRef(int row, int col)
    {
      int id = m_colPtrs[cols];
      int end = m_colPtrs[cols+1];
      while (id<end && m_data.index(id)!=row)
      {
        ++id;
      }
      ei_assert(id!=end);
      return m_data.value(id);
    }

  public:

    class InnerIterator;

    inline int rows() const { return _rows(); }
    inline int cols() const { return _cols(); }
    /** \returns the number of non zero coefficients */
    inline int nonZeros() const  { return m_data.size(); }

    inline const Scalar& operator() (int row, int col) const
    {
      return _coeff(row, col);
    }

    inline Scalar& operator() (int row, int col)
    {
      return _coeffRef(row, col);
    }

    inline void startFill(int reserveSize = 1000)
    {
      m_data.clear();
      m_data.reserve(reserveSize);
      for (int i=0; i<=m_cols; ++i)
        m_colPtrs[i] = 0;
    }

    inline Scalar& fill(int row, int col)
    {
      if (m_colPtrs[col+1]==0)
      {
        int i=col;
        while (i>=0 && m_colPtrs[i]==0)
        {
          m_colPtrs[i] = m_data.size();
          --i;
        }
        m_colPtrs[col+1] = m_colPtrs[col];
      }
      assert(m_colPtrs[col+1] == m_data.size());
      int id = m_colPtrs[col+1];
      m_colPtrs[col+1]++;

      m_data.append(0, row);
      return m_data.value(id);
    }

    inline void endFill()
    {
      int size = m_data.size();
      int i = m_cols;
      // find the last filled column
      while (i>=0 && m_colPtrs[i]==0)
        --i;
      i++;
      while (i<=m_cols)
      {
        m_colPtrs[i] = size;
        ++i;
      }
    }

    void resize(int rows, int cols)
    {
      if (m_cols != cols)
      {
        delete[] m_colPtrs;
        m_colPtrs = new int [cols+1];
        m_rows = rows;
        m_cols = cols;
      }
    }

    inline SparseMatrix(int rows, int cols)
      : m_rows(0), m_cols(0), m_colPtrs(0)
    {
      resize(rows, cols);
    }

    inline SparseMatrix& operator=(const SparseMatrix& other)
    {
      resize(other.rows(), other.cols());
      m_colPtrs = other.m_colPtrs;
      for (int col=0; col<=cols(); ++col)
        m_colPtrs[col] = other.m_colPtrs[col];
      m_data = other.m_data;
      return *this;
    }

    template<typename OtherDerived>
    inline SparseMatrix& operator=(const MatrixBase<OtherDerived>& other)
    {
      resize(other.rows(), other.cols());
      startFill(std::max(m_rows,m_cols)*2);
      for (int col=0; col<cols(); ++col)
      {
        for (typename OtherDerived::InnerIterator it(other.derived(), col); it; ++it)
        {
          Scalar v = it.value();
          if (v!=Scalar(0))
            fill(it.index(),col) = v;
        }
      }
      endFill();
      return *this;
    }


    // old explicit operator+
//     template<typename Other>
//     SparseMatrix operator+(const Other& other)
//     {
//       SparseMatrix res(rows(), cols());
//       res.startFill(nonZeros()*3);
//       for (int col=0; col<cols(); ++col)
//       {
//         InnerIterator row0(*this,col);
//         typename Other::InnerIterator row1(other,col);
//         while (row0 && row1)
//         {
//           if (row0.index()==row1.index())
//           {
//             std::cout << "both " << col << " " << row0.index() << "\n";
//             Scalar v = row0.value() + row1.value();
//             if (v!=Scalar(0))
//               res.fill(row0.index(),col) = v;
//             ++row0;
//             ++row1;
//           }
//           else if (row0.index()<row1.index())
//           {
//             std::cout << "row0 " << col << " " << row0.index() << "\n";
//             Scalar v = row0.value();
//             if (v!=Scalar(0))
//               res.fill(row0.index(),col) = v;
//             ++row0;
//           }
//           else if (row1)
//           {
//             std::cout << "row1 " << col << " " << row0.index() << "\n";
//             Scalar v = row1.value();
//             if (v!=Scalar(0))
//               res.fill(row1.index(),col) = v;
//             ++row1;
//           }
//         }
//         while (row0)
//         {
//           std::cout << "row0 " << col << " " << row0.index() << "\n";
//           Scalar v = row0.value();
//           if (v!=Scalar(0))
//             res.fill(row0.index(),col) = v;
//           ++row0;
//         }
//         while (row1)
//         {
//           std::cout << "row1 " << col << " " << row1.index() << "\n";
//           Scalar v = row1.value();
//           if (v!=Scalar(0))
//             res.fill(row1.index(),col) = v;
//           ++row1;
//         }
//       }
//       res.endFill();
//       return res;
// //       return binaryOp(other, ei_scalar_sum_op<Scalar>());
//     }


    // WARNING for efficiency reason it currently outputs the transposed matrix
    friend std::ostream & operator << (std::ostream & s, const SparseMatrix& m)
    {
      s << "Nonzero entries:\n";
      for (uint i=0; i<m.nonZeros(); ++i)
      {
        s << "(" << m.m_data.value(i) << "," << m.m_data.index(i) << ") ";
      }
      s << std::endl;
      s << std::endl;
      s << "Column pointers:\n";
      for (uint i=0; i<m.cols(); ++i)
      {
        s << m.m_colPtrs[i] << " ";
      }
      s << std::endl;
      s << std::endl;
      s << "Matrix (transposed):\n";
      for (int j=0; j<m.cols(); j++ )
      {
        int end = m.m_colPtrs[j+1];
        int i=0;
        for (int id=m.m_colPtrs[j]; id<end; id++)
        {
          int row = m.m_data.index(id);
          // fill with zeros
          for (int k=i; k<row; ++k)
            s << "0 ";
          i = row+1;
          s << m.m_data.value(id) << " ";
        }
        for (int k=i; k<m.rows(); ++k)
          s << "0 ";
        s << std::endl;
      }
      return s;
    }

    /** Destructor */
    inline ~SparseMatrix()
    {
      delete[] m_colPtrs;
    }
};

template<typename Scalar>
class SparseMatrix<Scalar>::InnerIterator
{
  public:
    InnerIterator(const SparseMatrix& mat, int col)
      : m_matrix(mat), m_id(mat.m_colPtrs[col]), m_start(m_id), m_end(mat.m_colPtrs[col+1])
    {}

    InnerIterator& operator++() { m_id++; return *this; }

    Scalar value() { return m_matrix.m_data.value(m_id); }

    int index() const { return m_matrix.m_data.index(m_id); }

    operator bool() const { return (m_id < m_end) && (m_id>=m_start); }

  protected:
    const SparseMatrix& m_matrix;
    int m_id;
    const int m_start;
    const int m_end;
};

#endif // EIGEN_SPARSEMATRIX_H
