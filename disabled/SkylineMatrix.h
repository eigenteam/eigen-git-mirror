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
  * \brief 
  *
  * \param
  *
  * \sa 
  */
template<typename _Scalar, int Size, int Supers, int Subs, int Options>
struct ei_traits<BandMatrix<_Scalar,Size,Supers,Subs,Options> >
{
  typedef _Scalar Scalar;
  enum {
    CoeffReadCost = NumTraits<Scalar>::ReadCost,
    RowsAtCompileTime = Size,
    ColsAtCompileTime = Size,
    MaxRowsAtCompileTime = Size,
    MaxColsAtCompileTime = Size,
    Flags = 0
  };
};

template<typename _Scalar, int Size, int Supers, int Subs, int Options>
class BandMatrix : public MultiplierBase<BandMatrix<_Scalar,Supers,Subs,Options> >
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
      DataSizeAtCompileTime = ((Size!=Dynamic) && (Supers!=Dynamic) && (Subs!=Dynamic))
                            ? Size*(Supers+Subs+1) - (Supers*Supers+Subs*Subs)/2
                            : Dynamic
    };
    typedef Matrix<Scalar,DataSizeAtCompileTime,1> DataType;
    
  public:

//     inline BandMatrix() { }

    inline BandMatrix(int size=Size, int supers=Supers, int subs=Subs)
      : m_data(size*(supers+subs+1) - (supers*supers+subs*subs)/2),
        m_size(size), m_supers(supers), m_subs(subs)
    { }

    inline int rows() const { return m_size.value(); }
    inline int cols() const { return m_size.value(); }

    inline int supers() const { return m_supers.value(); }
    inline int subs() const { return m_subs.value(); }

    inline VectorBlock<DataType,Size> diagonal()
    { return VectorBlock<DataType,Size>(m_data,0,m_size.value()); }

    inline const VectorBlock<DataType,Size> diagonal() const
    { return VectorBlock<DataType,Size>(m_data,0,m_size.value()); }

    template<int Index>
    VectorBlock<DataType,Size==Dynamic?Dynamic:Size-(Index<0?-Index:Index)>
    diagonal()
    {
      return VectorBlock<DataType,Size==Dynamic?Dynamic:Size-(Index<0?-Index:Index)>
        (m_data,Index<0 ? subDiagIndex(-Index) : superDiagIndex(Index), m_size.value()-ei_abs(Index));
    }

    template<int Index>
    const VectorBlock<DataType,Size==Dynamic?Dynamic:Size-(Index<0?-Index:Index)>
    diagonal() const
    {
      return VectorBlock<DataType,Size==Dynamic?Dynamic:Size-(Index<0?-Index:Index)>
        (m_data,Index<0 ? subDiagIndex(-Index) : superDiagIndex(Index), m_size.value()-ei_abs(Index));
    }

    inline VectorBlock<DataType,Dynamic> diagonal(int index)
    {
      ei_assert((index<0 && -index<=subs()) || (index>=0 && index<=supers()));
      return VectorBlock<DataType,Dynamic>(m_data,
              index<0 ? subDiagIndex(-index) : superDiagIndex(index), m_size.value()-ei_abs(index));
    }
    const VectorBlock<DataType,Dynamic> diagonal(int index) const
    {
      ei_assert((index<0 && -index<=subs()) || (index>=0 && index<=supers()));
      return VectorBlock<DataType,Dynamic>(m_data,
              index<0 ? subDiagIndex(-index) : superDiagIndex(index), m_size.value()-ei_abs(index));
    }

//     inline VectorBlock<DataType,Size> subDiagonal()
//     { return VectorBlock<DataType,Size>(m_data,0,m_size.value()); }

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

    inline int subDiagIndex(int i) const
    { return m_size.value()*(m_supers.value()+i)-(ei_abs2(i-1) + ei_abs2(m_supers.value()))/2; }

    inline int superDiagIndex(int i) const
    { return m_size.value()*i-ei_abs2(i-1)/2; }

    DataType m_data;
    ei_int_if_dynamic<Size>   m_size;
    ei_int_if_dynamic<Supers> m_supers;
    ei_int_if_dynamic<Subs>   m_subs;
};

#endif // EIGEN_BANDMATRIX_H
