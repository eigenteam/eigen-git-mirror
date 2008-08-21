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

#ifndef EIGEN_IO_H
#define EIGEN_IO_H

enum { Raw, AlignCols };

struct IoFormat
{
  IoFormat(int _precision=4, int _flags=Raw,
    const std::string& _coeffSeparator = " ",
    const std::string& _rowSeparator = "\n", const std::string& _rowPrefix="", const std::string& _rowSuffix="",
    const std::string& _matPrefix="", const std::string& _matSuffix="")
  : matPrefix(_matPrefix), matSuffix(_matSuffix), rowPrefix(_rowPrefix), rowSuffix(_rowSuffix), rowSeparator(_rowSeparator),
    coeffSeparator(_coeffSeparator), precision(_precision), flags(_flags)
  {
    rowSpacer = "";
    int i=matSuffix.length()-1;
    while (i>=0 && matSuffix[i]!='\n')
    {
      rowSpacer += ' ';
      i--;
    }
  }
  std::string matPrefix, matSuffix;
  std::string rowPrefix, rowSuffix, rowSeparator, rowSpacer;
  std::string coeffSeparator;
  int precision;
  int flags;
};

template<typename ExpressionType>
class WithFormat
{
  public:

    WithFormat(const ExpressionType& matrix, const IoFormat& format)
      : m_matrix(matrix), m_format(format)
    {}

    friend std::ostream & operator << (std::ostream & s, const WithFormat& wf)
    {
      return ei_print_matrix(s, wf.m_matrix.eval(), wf.m_format);
    }

  protected:
    const typename ExpressionType::Nested m_matrix;
    IoFormat m_format;
};

template<typename Derived>
inline const WithFormat<Derived>
MatrixBase<Derived>::format(const IoFormat& fmt) const
{
  return WithFormat<Derived>(derived(), fmt);
}

template<typename Derived>
std::ostream & ei_print_matrix(std::ostream & s, const MatrixBase<Derived> & _m,
                               const IoFormat& fmt = IoFormat())
{
  const typename Derived::Nested m = _m;
  int width = 0;
  if (fmt.flags & AlignCols)
  {
    // compute the largest width
    for(int j = 1; j < m.cols(); j++)
      for(int i = 0; i < m.rows(); i++)
      {
        std::stringstream sstr;
        sstr.precision(fmt.precision);
        sstr << m.coeff(i,j);
        width = std::max<int>(width, sstr.str().length());
      }
  }
  s.precision(fmt.precision);
  s << fmt.matPrefix;
  for(int i = 0; i < m.rows(); i++)
  {
    if (i)
      s << fmt.rowSpacer;
    s << fmt.rowPrefix;
    if(width) s.width(width);
    s << m.coeff(i, 0);
    for(int j = 1; j < m.cols(); j++)
    {
      s << fmt.coeffSeparator;
      if (width) s.width(width);
      s << m.coeff(i, j);
    }
    s << fmt.rowSuffix;
    if( i < m.rows() - 1)
      s << fmt.rowSeparator;
  }
  s << fmt.matSuffix;
  return s;
}

/** \relates MatrixBase
  *
  * Outputs the matrix, laid out as an array as usual, to the given stream.
  */
template<typename Derived>
std::ostream & operator <<
(std::ostream & s,
 const MatrixBase<Derived> & m)
{
  return ei_print_matrix(s, m.eval());
}

#endif // EIGEN_IO_H
