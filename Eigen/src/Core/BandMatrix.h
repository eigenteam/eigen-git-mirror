// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_BANDMATRIX_H
#define EIGEN_BANDMATRIX_H

/** \nonstableyet
  * \class BandMatrix
  *
  * \brief Represents a rectangular matrix with a banded storage
  *
  * \param _Scalar Numeric type, i.e. float, double, int
  * \param Rows Number of rows, or \b Dynamic
  * \param Cols Number of columns, or \b Dynamic
  * \param Supers Number of super diagonal
  * \param Subs Number of sub diagonal
  * \param _Options A combination of either \b RowMajor or \b ColMajor, and of \b SelfAdjoint
  *                 The former controls storage order, and defaults to column-major. The latter controls
  *                 whether the matrix represent a selfadjoint matrix in which case either Supers of Subs
  *                 have to be null.
  *
  * \sa class TridiagonalMatrix
  */
template<typename _Scalar, int Rows, int Cols, int Supers, int Subs, int Options>
struct ei_traits<BandMatrix<_Scalar,Rows,Cols,Supers,Subs,Options> >
{
  typedef _Scalar Scalar;
  enum {
    CoeffReadCost = NumTraits<Scalar>::ReadCost,
    RowsAtCompileTime = Rows,
    ColsAtCompileTime = Cols,
    MaxRowsAtCompileTime = Rows,
    MaxColsAtCompileTime = Cols,
    Flags = 0
  };
};

template<typename _Scalar, int Rows, int Cols, int Supers, int Subs, int Options>
class BandMatrix : public AnyMatrixBase<BandMatrix<_Scalar,Rows,Cols,Supers,Subs,Options> >
{
  public:

    enum {
      Flags = ei_traits<BandMatrix>::Flags,
      CoeffReadCost = ei_traits<BandMatrix>::CoeffReadCost,
      RowsAtCompileTime = ei_traits<BandMatrix>::RowsAtCompileTime,
      ColsAtCompileTime = ei_traits<BandMatrix>::ColsAtCompileTime,
      MaxRowsAtCompileTime = ei_traits<BandMatrix>::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = ei_traits<BandMatrix>::MaxColsAtCompileTime
    };
    typedef typename ei_traits<BandMatrix>::Scalar Scalar;
    typedef Matrix<Scalar,RowsAtCompileTime,ColsAtCompileTime> PlainMatrixType;

  protected:
    enum {
      DataRowsAtCompileTime = ((Supers!=Dynamic) && (Subs!=Dynamic))
                            ? 1 + Supers + Subs
                            : Dynamic,
      SizeAtCompileTime = EIGEN_ENUM_MIN(Rows,Cols)
    };
    typedef Matrix<Scalar,DataRowsAtCompileTime,ColsAtCompileTime,Options&RowMajor?RowMajor:ColMajor> DataType;

  public:

    inline BandMatrix(int rows=Rows, int cols=Cols, int supers=Supers, int subs=Subs)
      : m_data(1+supers+subs,cols),
        m_rows(rows), m_supers(supers), m_subs(subs)
    {
        m_data.setConstant(666);
    }

    /** \returns the number of columns */
    inline int rows() const { return m_rows.value(); }

    /** \returns the number of rows */
    inline int cols() const { return m_data.cols(); }

    /** \returns the number of super diagonals */
    inline int supers() const { return m_supers.value(); }

    /** \returns the number of sub diagonals */
    inline int subs() const { return m_subs.value(); }

    /** \returns a vector expression of the \a i -th column,
      * only the meaningful part is returned.
      * \warning the internal storage must be column major. */
    inline Block<DataType,Dynamic,1> col(int i)
    {
      EIGEN_STATIC_ASSERT((Options&RowMajor)==0,THIS_METHOD_IS_ONLY_FOR_COLUMN_MAJOR_MATRICES);
      int start = 0;
      int len = m_data.rows();
      if (i<=supers())
      {
        start = supers()-i;
        len = std::min(rows(),std::max(0,m_data.rows() - (supers()-i)));
      }
      else if (i>=rows()-subs())
        len = std::max(0,m_data.rows() - (i + 1 - rows() + subs()));
      return Block<DataType,Dynamic,1>(m_data, start, i, len, 1);
    }

    /** \returns a vector expression of the main diagonal */
    inline Block<DataType,1,SizeAtCompileTime> diagonal()
    { return Block<DataType,1,SizeAtCompileTime>(m_data,supers(),0,1,std::min(rows(),cols())); }

    /** \returns a vector expression of the main diagonal (const version) */
    inline const Block<DataType,1,SizeAtCompileTime> diagonal() const
    { return Block<DataType,1,SizeAtCompileTime>(m_data,supers(),0,1,std::min(rows(),cols())); }

    template<int Index> struct DiagonalIntReturnType {
      enum {
        ReturnOpposite = (Options&SelfAdjoint) && (Index>0 && Supers==0 || Index<0 && Subs==0),
        Conjugate = ReturnOpposite && NumTraits<Scalar>::IsComplex,
        ActualIndex = ReturnOpposite ? -Index : Index,
        DiagonalSize = RowsAtCompileTime==Dynamic || ColsAtCompileTime==Dynamic
                     ? Dynamic
                     : ActualIndex<0
                     ? EIGEN_ENUM_MIN(ColsAtCompileTime, RowsAtCompileTime + ActualIndex)
                     : EIGEN_ENUM_MIN(RowsAtCompileTime, ColsAtCompileTime - ActualIndex)
      };
      typedef Block<DataType,1, DiagonalSize> BuildType;
      typedef typename ei_meta_if<Conjugate,
                 CwiseUnaryOp<ei_scalar_conjugate_op<Scalar>,NestByValue<BuildType> >,
                 BuildType>::ret Type;
    };

    /** \returns a vector expression of the \a Index -th sub or super diagonal */
    template<int Index> inline typename DiagonalIntReturnType<Index>::Type diagonal()
    {
      return typename DiagonalIntReturnType<Index>::BuildType(m_data, supers()-Index, std::max(0,Index), 1, diagonalLength(Index));
    }

    /** \returns a vector expression of the \a Index -th sub or super diagonal */
    template<int Index> inline const typename DiagonalIntReturnType<Index>::Type diagonal() const
    {
      return typename DiagonalIntReturnType<Index>::BuildType(m_data, supers()-Index, std::max(0,Index), 1, diagonalLength(Index));
    }

    /** \returns a vector expression of the \a i -th sub or super diagonal */
    inline Block<DataType,1,Dynamic> diagonal(int i)
    {
      ei_assert((i<0 && -i<=subs()) || (i>=0 && i<=supers()));
      return Block<DataType,1,Dynamic>(m_data, supers()-i, std::max(0,i), 1, diagonalLength(i));
    }

    /** \returns a vector expression of the \a i -th sub or super diagonal */
    inline const Block<DataType,1,Dynamic> diagonal(int i) const
    {
      ei_assert((i<0 && -i<=subs()) || (i>=0 && i<=supers()));
      return Block<DataType,1,Dynamic>(m_data, supers()-i, std::max(0,i), 1, diagonalLength(i));
    }

    PlainMatrixType toDense() const
    {
      PlainMatrixType res(rows(),cols());
      res.setZero();
      res.diagonal() = diagonal();
      for (int i=1; i<=supers();++i)
        res.diagonal(i) = diagonal(i);
      for (int i=1; i<=subs();++i)
        res.diagonal(-i) = diagonal(-i);
      return res;
    }

  protected:

    inline int diagonalLength(int i) const
    { return i<0 ? std::min(cols(),rows()+i) : std::min(rows(),cols()-i); }

    DataType m_data;
    ei_int_if_dynamic<Rows>   m_rows;
    ei_int_if_dynamic<Supers> m_supers;
    ei_int_if_dynamic<Subs>   m_subs;
};

/** \nonstableyet
  * \class TridiagonalMatrix
  *
  * \brief Represents a tridiagonal matrix
  *
  * \param _Scalar Numeric type, i.e. float, double, int
  * \param Size Number of rows and cols, or \b Dynamic
  * \param _Options Can be 0 or \b SelfAdjoint
  *
  * \sa class BandMatrix
  */
template<typename Scalar, int Size, int Options>
class TridiagonalMatrix : public BandMatrix<Scalar,Size,Size,Options&SelfAdjoint?0:1,1,Options|RowMajor>
{
    typedef BandMatrix<Scalar,Size,Size,1,Options&SelfAdjoint?0:1,Options|RowMajor> Base;
  public:
    TridiagonalMatrix(int size = Size) : Base(size,size,1,1) {}

    inline typename Base::template DiagonalIntReturnType<1>::Type super()
    { return Base::template diagonal<1>(); }
    inline const typename Base::template DiagonalIntReturnType<1>::Type super() const
    { return Base::template diagonal<1>(); }
    inline typename Base::template DiagonalIntReturnType<-1>::Type sub()
    { return Base::template diagonal<-1>(); }
    inline const typename Base::template DiagonalIntReturnType<-1>::Type sub() const
    { return Base::template diagonal<-1>(); }
  protected:
};

#endif // EIGEN_BANDMATRIX_H
