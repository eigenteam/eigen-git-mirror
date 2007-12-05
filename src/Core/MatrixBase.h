// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Eigen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#ifndef EIGEN_MATRIXBASE_H
#define EIGEN_MATRIXBASE_H

template<typename Scalar, typename Derived> class MatrixBase
{
    static const int RowsAtCompileTime = Derived::RowsAtCompileTime,
                     ColsAtCompileTime = Derived::ColsAtCompileTime;
    
    template<typename OtherDerived>
    void _copy_helper(const MatrixBase<Scalar, OtherDerived>& other);
    
  public:
    static const int SizeAtCompileTime
      = RowsAtCompileTime == Dynamic || ColsAtCompileTime == Dynamic
      ? Dynamic : RowsAtCompileTime * ColsAtCompileTime;
    static const bool IsVector = RowsAtCompileTime == 1 || ColsAtCompileTime == 1;
    
    typedef typename ForwardDecl<Derived>::Ref Ref;
    typedef typename NumTraits<Scalar>::Real RealScalar;
  
    int rows() const { return static_cast<const Derived *>(this)->_rows(); }
    int cols() const { return static_cast<const Derived *>(this)->_cols(); }
    int size() const { return rows() * cols(); }
    
    Ref ref() const
    { return static_cast<const Derived *>(this)->_ref(); }
    
    Scalar& write(int row, int col)
    {
      // from this single point, we can check all writes to all matrix/expression
      // types. We only want this however for internal testing, as this is very slow.
      eigen_internal_assert(row >= 0 && row < rows()
                         && col >= 0 && col < cols());
      return static_cast<Derived *>(this)->_write(row, col);
    }
    
    Scalar read(int row, int col) const
    {
      // from this single point, we can check all reads to all matrix/expression
      // types. We only want this however for internal testing, as this is very slow.
      eigen_internal_assert(row >= 0 && row < rows()
                         && col >= 0 && col < cols());
      return static_cast<const Derived *>(this)->_read(row, col);
    }
    
    template<typename OtherDerived>
    Derived& operator=(const MatrixBase<Scalar, OtherDerived>& other)
    {
      assert(rows() == other.rows() && cols() == other.cols());
      _copy_helper(other);
      return *static_cast<Derived*>(this);
    }
    
    //special case of the above template operator=, in order to prevent the compiler
    //from generating a default operator= (issue hit with g++ 4.1)
    Derived& operator=(const MatrixBase& other)
    {
      assert(rows() == other.rows() && cols() == other.cols());
      _copy_helper(other);
      return *static_cast<Derived*>(this);
    }
    
    template<typename NewScalar> Cast<NewScalar, Derived> cast() const;
    
    Row<Derived> row(int i) const;
    Column<Derived> col(int i) const;
    Minor<Derived> minor(int row, int col) const;
    
    DynBlock<Derived> dynBlock(int startRow, int startCol,
                               int blockRows, int blockCols) const;
    template<int BlockRows, int BlockCols>
    Block<Derived, BlockRows, BlockCols> block(int startRow, int startCol) const;
    
    Transpose<Derived> transpose() const;
    Conjugate<Derived> conjugate() const;
    Transpose<Conjugate<Derived> > adjoint() const { return conjugate().transpose(); }
    Scalar trace() const;
    
    template<typename OtherDerived>
    Scalar dot(const OtherDerived& other) const;
    RealScalar norm2() const;
    RealScalar norm()  const;
    ScalarMultiple<Derived> normalized() const;
    
    static Eval<Random<Derived> >
    random(int rows = RowsAtCompileTime, int cols = ColsAtCompileTime);
    static Zero<Derived>
    zero(int rows = RowsAtCompileTime, int cols = ColsAtCompileTime);
    static Identity<Derived>
    identity(int rows = RowsAtCompileTime);
    static FromArray<Derived>
    fromArray(const Scalar* array, int rows = RowsAtCompileTime, int cols = ColsAtCompileTime);
    
    template<typename OtherDerived>
    bool isApprox(
      const OtherDerived& other,
      const typename NumTraits<Scalar>::Real& prec = precision<Scalar>()
    ) const;
    bool isMuchSmallerThan(
      const typename NumTraits<Scalar>::Real& other,
      const typename NumTraits<Scalar>::Real& prec = precision<Scalar>()
    ) const;
    template<typename OtherDerived>
    bool isMuchSmallerThan(
      const MatrixBase<Scalar, OtherDerived>& other,
      const typename NumTraits<Scalar>::Real& prec = precision<Scalar>()
    ) const;
    
    template<typename OtherDerived>
    Product<Derived, OtherDerived>
    lazyProduct(const MatrixBase<Scalar, OtherDerived>& other) const EIGEN_ALWAYS_INLINE;
    
    Opposite<Derived> operator-() const;
    
    template<typename OtherDerived>
    Derived& operator+=(const MatrixBase<Scalar, OtherDerived>& other);
    template<typename OtherDerived>
    Derived& operator-=(const MatrixBase<Scalar, OtherDerived>& other);
    template<typename OtherDerived>
    Derived& operator*=(const MatrixBase<Scalar, OtherDerived>& other);
   
    Derived& operator*=(const int& other);
    Derived& operator*=(const float& other);
    Derived& operator*=(const double& other);
    Derived& operator*=(const std::complex<float>& other);
    Derived& operator*=(const std::complex<double>& other);
    
    Derived& operator/=(const int& other);
    Derived& operator/=(const float& other);
    Derived& operator/=(const double& other);
    Derived& operator/=(const std::complex<float>& other);
    Derived& operator/=(const std::complex<double>& other);

    Scalar operator()(int row, int col) const
    {
      assert(row >= 0 && row < rows()
          && col >= 0 && col < cols());
      return read(row, col);
    }
    
    Scalar& operator()(int row, int col)
    {
      assert(row >= 0 && row < rows()
          && col >= 0 && col < cols());
      return write(row, col);
    }
    
    Scalar operator[](int index) const
    {
      assert(IsVector);
      if(RowsAtCompileTime == 1)
      {
        assert(index >= 0 && index < cols());
        return read(0, index);
      }
      else
      {
        assert(index >= 0 && index < rows());
        return read(index, 0);
      }
    }
    
    Scalar& operator[](int index)
    {
      assert(IsVector);
      if(RowsAtCompileTime == 1)
      {
        assert(index >= 0 && index < cols());
        return write(0, index);
      }
      else
      {
        assert(index >= 0 && index < rows());
        return write(index, 0);
      }
    }
    
    Eval<Derived> eval() const EIGEN_ALWAYS_INLINE;
};

template<typename Scalar, typename Derived>
std::ostream & operator <<
( std::ostream & s,
  const MatrixBase<Scalar, Derived> & m )
{
  for( int i = 0; i < m.rows(); i++ )
  {
    s << m( i, 0 );
    for (int j = 1; j < m.cols(); j++ )
      s << " " << m( i, j );
    if( i < m.rows() - 1)
      s << std::endl;
  }
  return s;
}

#endif // EIGEN_MATRIXBASE_H
