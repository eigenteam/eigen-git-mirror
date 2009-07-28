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

#ifndef EIGEN_RETURNBYVALUE_H
#define EIGEN_RETURNBYVALUE_H

/** \class ReturnByValue
  *
  */
template<typename Functor, typename _Scalar,int _Rows,int _Cols,int _Options,int _MaxRows,int _MaxCols>
struct ei_traits<ReturnByValue<Functor,Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> > >
  : public ei_traits<Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> >
{
  enum {
    Flags = ei_traits<Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> >::Flags | EvalBeforeNestingBit
  };
};

template<typename Functor,typename EvalTypeDerived,int n>
struct ei_nested<ReturnByValue<Functor,MatrixBase<EvalTypeDerived> >, n, EvalTypeDerived>
{
  typedef EvalTypeDerived type;
};

template<typename Functor, typename EvalType> class ReturnByValue
{
  public:
    template<typename Dest> inline void evalTo(Dest& dst) const
    { static_cast<const Functor*>(this)->evalTo(dst); }
    template<typename Dest> inline void addTo(Dest& dst) const
    { static_cast<const Functor*>(this)->_addTo(dst); }
    template<typename Dest> inline void subTo(Dest& dst) const
    { static_cast<const Functor*>(this)->_subTo(dst); }
    template<typename Dest> inline void _addTo(Dest& dst) const
    { EvalType res; evalTo(res); dst += res; }
    template<typename Dest> inline void _subTo(Dest& dst) const
    { EvalType res; evalTo(res); dst -= res; }
};

template<typename Functor, typename _Scalar,int _Rows,int _Cols,int _Options,int _MaxRows,int _MaxCols>
  class ReturnByValue<Functor,Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> >
  : public MatrixBase<ReturnByValue<Functor,Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> > >
{
    typedef Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> EvalType;
  public:
    EIGEN_GENERIC_PUBLIC_INTERFACE(ReturnByValue)
    template<typename Dest>
    inline void evalTo(Dest& dst) const
    { static_cast<const Functor* const>(this)->evalTo(dst); }
    template<typename Dest> inline void addTo(Dest& dst) const
    { static_cast<const Functor*>(this)->_addTo(dst); }
    template<typename Dest> inline void subTo(Dest& dst) const
    { static_cast<const Functor*>(this)->_subTo(dst); }
    template<typename Dest> inline void _addTo(Dest& dst) const
    { EvalType res; evalTo(res); dst += res; }
    template<typename Dest> inline void _subTo(Dest& dst) const
    { EvalType res; evalTo(res); dst -= res; }
    inline int rows() const { return static_cast<const Functor* const>(this)->rows(); }
    inline int cols() const { return static_cast<const Functor* const>(this)->cols(); }
};

template<typename Derived>
template<typename OtherDerived,typename OtherEvalType>
Derived& MatrixBase<Derived>::operator=(const ReturnByValue<OtherDerived,OtherEvalType>& other)
{
  other.evalTo(derived());
  return derived();
}

template<typename Derived>
template<typename OtherDerived,typename OtherEvalType>
Derived& MatrixBase<Derived>::operator+=(const ReturnByValue<OtherDerived,OtherEvalType>& other)
{
  other.addTo(derived());
  return derived();
}

template<typename Derived>
template<typename OtherDerived,typename OtherEvalType>
Derived& MatrixBase<Derived>::operator-=(const ReturnByValue<OtherDerived,OtherEvalType>& other)
{
  other.subTo(derived());
  return derived();
}

#endif // EIGEN_RETURNBYVALUE_H
