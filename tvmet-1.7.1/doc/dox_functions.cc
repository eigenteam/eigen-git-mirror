/*
 * Tiny Vector Matrix Library
 * Dense Vector Matrix Libary of Tiny size using Expression Templates
 *
 * Copyright (C) 2001 - 2003 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: dox_functions.cc,v 1.8 2003/12/19 18:01:37 opetzold Exp $
 */

#include <iostream>

#include <tvmet/config.h>

#include "Util.h"

class FunctionBase
{
public:
  FunctionBase() {
    m_binary_functions.push_back( BinaryFunction("atan2", "arcus tangent of two variables") );
    m_binary_functions.push_back( BinaryFunction("drem", "floating-point remainder") );
    m_binary_functions.push_back( BinaryFunction("fmod", "floating-point remainder") );
    m_binary_functions.push_back( BinaryFunction("hypot", "Euclidean distance") );
    m_binary_functions.push_back( BinaryFunction("jn", "Bessel") );
    m_binary_functions.push_back( BinaryFunction("yn", "Bessel") );
    m_binary_functions.push_back( BinaryFunction("pow", "power") );

    m_unary_functions.push_back( UnaryFunction("abs", "absolute value") );
    m_unary_functions.push_back( UnaryFunction("cbrt", "cube root") );
    m_unary_functions.push_back( UnaryFunction("floor", "round") );
    m_unary_functions.push_back( UnaryFunction("rint", "round") );
    m_unary_functions.push_back( UnaryFunction("sin", "sin") );
    m_unary_functions.push_back( UnaryFunction("sinh", "sinh") );
    m_unary_functions.push_back( UnaryFunction("cos", "cos") );
    m_unary_functions.push_back( UnaryFunction("cosh", "cosh") );
    m_unary_functions.push_back( UnaryFunction("asin", "asin") );
    m_unary_functions.push_back( UnaryFunction("acos", "acos") );
    m_unary_functions.push_back( UnaryFunction("atan", "atan") );
    m_unary_functions.push_back( UnaryFunction("exp", "exponential") );
    m_unary_functions.push_back( UnaryFunction("log", "logarithmic") );
    m_unary_functions.push_back( UnaryFunction("log10", "logarithmic") );
    m_unary_functions.push_back( UnaryFunction("sqrt", "sqrt") );
#ifdef TVMET_HAVE_IEEE_MATH
    m_unary_functions.push_back( UnaryFunction("asinh", "IEEE Math asinh") );
    m_unary_functions.push_back( UnaryFunction("acosh", "IEEE Math acosh") );
    m_unary_functions.push_back( UnaryFunction("atanh", "IEEE Math atanh") );
    m_unary_functions.push_back( UnaryFunction("expm1", "IEEE Math expm1") );
    m_unary_functions.push_back( UnaryFunction("log1p", "IEEE Math log1p") );
    m_unary_functions.push_back( UnaryFunction("erf", "IEEE Math erf") );
    m_unary_functions.push_back( UnaryFunction("erfc", "IEEE Math erfc") );
    m_unary_functions.push_back( UnaryFunction("isnan", "IEEE Math isnan. "
					       "Return nonzero value if X is a NaN.") );
    m_unary_functions.push_back( UnaryFunction("isinf", "IEEE Math isinf. "
					       "Return nonzero value if X is positive or negative infinity.") );
    m_unary_functions.push_back( UnaryFunction("isfinite", "fIEEE Math isfinite. "
					       "Return nonzero value if X is not +-Inf or NaN.") );
    m_unary_functions.push_back( UnaryFunction("j0", "IEEE Math Bessel") );
    m_unary_functions.push_back( UnaryFunction("j1", "IEEE Math Bessel") );
    m_unary_functions.push_back( UnaryFunction("y0", "IEEE Math Bessel") );
    m_unary_functions.push_back( UnaryFunction("y1", "IEEE Math Bessel") );
    m_unary_functions.push_back( UnaryFunction("lgamma", "IEEE Math lgamma") );
#endif
  }

  virtual ~FunctionBase() { }

public:
  template<class Stream>
  Stream& header(Stream& os) const {
    m_type.header(os);
    return os;
  }

  template<class Stream>
  Stream& footer(Stream& os) const {
    m_type.footer(os);
    return os;
  }

  template<class Stream>
  Stream& binary(Stream& os) const {
    return os;
  }

  template<class Stream>
  Stream& unary(Stream& os) const {
    return os;
  }

public:
  typedef std::vector< BinaryFunction >::const_iterator	bfun_iterator;
  typedef std::vector< UnaryFunction >::const_iterator	ufun_iterator;

public:
  virtual const std::vector< BinaryFunction >& bfun() const { return m_binary_functions; }
  virtual const std::vector< UnaryFunction >& ufun() const { return m_unary_functions; }

protected:
  std::vector< BinaryFunction >		m_binary_functions;
  std::vector< UnaryFunction >		m_unary_functions;
  Type					m_type;
};



class XprFunctions : public FunctionBase
{
public:
  XprFunctions() { }

public:
  template<class Stream>
  Stream& operator()(Stream& os) const {
    header(os);

    binary(os);
    binary2(os);
    unary(os);

    footer(os);

    return os;
  }

public:
  template<class Stream>
  Stream& header(Stream& os) const {
    FunctionBase::header(os);
    os << "//\n"
       << "// XprFunctions.h\n"
       << "//\n\n";
    return os;
  }

  // binary functions
  template<class Stream>
  Stream& binary(Stream& os) const {
    FunctionBase::binary(os);

    for(bfun_iterator fun = bfun().begin(); fun != bfun().end(); ++fun) {
      os << "/**\n"
	 << " * \\fn " << fun->name() << "(const XprVector<E1, Sz>& lhs, const XprVector<E2, Sz>& rhs)\n"
	 << " * \\brief " << fun->description() << " function for two XprVector.\n"
	 << " * \\ingroup " << fun->group() << "\n"
	 << " */\n\n";

      os << "/**\n"
	 << " * \\fn " << fun->name() << "(const XprMatrix<E1, Rows, Cols>& lhs, const XprMatrix<E2, Rows, Cols>& rhs)\n"
	 << " * \\brief " << fun->description() << " function for two XprMatrix.\n"
	 << " * \\ingroup " << fun->group() << "\n"
	 << " */\n\n";
    }
    return os;
  }

  // binary functions with pod and std::complex<>
  template<class Stream>
  Stream& binary2(Stream& os) const {
    for(Type::const_iterator tp = m_type.begin(); tp != m_type.end(); ++tp) {
      for(bfun_iterator fun = bfun().begin(); fun != bfun().end(); ++fun) {
	os << "/**\n"
	   << " * \\fn " << fun->name() << "(const XprVector<E, Sz>& lhs, " << tp->name() << " rhs)\n"
	   << " * \\brief " << fun->description() << " function between XprVector and " << tp->description() << ".\n"
	   << " * \\ingroup " << fun->group() << "\n"
	   << " */\n\n";

// 	os << "/**\n"
// 	   << " * \\fn " << fun->name() << "(" << tp->name() << " lhs, const XprVector<E, Sz>& rhs)\n"
// 	   << " * \\brief " << fun->description() << " function between " << tp->description() << " and XprVector.\n"
// 	   << " * \\ingroup " << fun->group() << "\n"
// 	   << " */\n\n";

	os << "/**\n"
	   << " * \\fn " << fun->name() << "(const XprMatrix<E, Rows, Cols>& lhs, " << tp->name() << " rhs)\n"
	   << " * \\brief " << fun->description() << " function between XprMatrix and " << tp->description() << ".\n"
	   << " * \\ingroup " << fun->group() << "\n"
	   << " */\n\n";

// 	os << "/**\n"
// 	   << " * \\fn " << fun->name() << "(" << tp->name() << " lhs, const XprMatrix<E, Rows, Cols>& rhs)\n"
// 	   << " * \\brief " << fun->description() << " function between " << tp->description() << " and XprMatrix.\n"
// 	   << " * \\ingroup " << fun->group() << "\n"
// 	   << " */\n\n";
      }
    }
    return os;
  }

  // unary functions
  template<class Stream>
  Stream& unary(Stream& os) const {
    FunctionBase::unary(os);

    for(ufun_iterator fun = ufun().begin(); fun != ufun().end(); ++fun) {
      os << "/**\n"
	 << " * \\fn " << fun->name() << "(const XprVector<E, Sz>& rhs)\n"
	 << " * \\brief " << fun->description() << " function for XprVector\n"
	 << " * \\ingroup " << fun->group() << "\n"
	 << " */\n\n";

      os << "/**\n"
	 << " * \\fn " << fun->name() << "(const XprMatrix<E, Rows, Cols>& rhs)\n"
	 << " * \\brief " << fun->description() << " function for XprMatrix.\n"
	 << " * \\ingroup " << fun->group() << "\n"
	 << " */\n\n";
    }
    return os;
  }
};




class VectorFunctions : public FunctionBase
{
public:
  VectorFunctions() { }

public:
  template<class Stream>
  Stream& operator()(Stream& os) const {
    header(os);

    binary(os);
    binary2(os);
    binary3(os);
    unary(os);

    footer(os);

    return os;
  }

public:
  template<class Stream>
  Stream& header(Stream& os) const {
    FunctionBase::header(os);

    os << "//\n"
       << "// VectorFunctions.h\n"
       << "//\n\n";
    return os;
  }

  // binary functions
  template<class Stream>
  Stream& binary(Stream& os) const {
    FunctionBase::binary(os);

    for(bfun_iterator fun = bfun().begin(); fun != bfun().end(); ++fun) {
      os << "/**\n"
	 << " * \\fn " << fun->name() << "(const Vector<T1, Sz>& lhs, const Vector<T2, Sz>& rhs)\n"
	 << " * \\brief " << fun->description() << " function for two Vector.\n"
	 << " * \\ingroup " << fun->group() << "\n"
	 << " */\n\n";
    }
    return os;
  }

  // binary functions with pod and std::complex<>
  template<class Stream>
  Stream& binary2(Stream& os) const {
    for(Type::const_iterator tp = m_type.begin(); tp != m_type.end(); ++tp) {
      for(bfun_iterator fun = bfun().begin(); fun != bfun().end(); ++fun) {
	os << "/**\n"
	   << " * \\fn " << fun->name() << "(const Vector<T, Sz>& lhs, " << tp->name() << " rhs)\n"
	   << " * \\brief " << fun->description() << " function on Vector and " << tp->description() << ".\n"
	   << " * \\ingroup " << fun->group() << "\n"
	   << " */\n\n";

// 	os << "/**\n"
// 	   << " * \\fn " << fun->name() << "(" << tp->name() << " lhs, const Vector<T, Sz>& rhs)\n"
// 	   << " * \\brief " << fun->description() << " function on " << tp->description() << " and Vector.\n"
// 	   << " * \\ingroup " << fun->group() << "\n"
// 	   << " */\n\n";
      }
    }
    return os;
  }

  // binary functions with expressions
  template<class Stream>
  Stream& binary3(Stream& os) const {
    for(bfun_iterator fun = bfun().begin(); fun != bfun().end(); ++fun) {
      os << "/**\n"
	 << " * \\fn " << fun->name() << "(const XprVector<E, Sz>& lhs, const Vector<T, Sz>& rhs)\n"
	 << " * \\brief " << fun->description() << " function on XprVector and Vector.\n"
	 << " * \\ingroup " << fun->group() << "\n"
	 << " */\n\n";

      os << "/**\n"
	 << " * \\fn " << fun->name() << "(const Vector<T, Sz>& lhs, const XprVector<E, Sz>& rhs)\n"
	 << " * \\brief " << fun->description() << " function on Vector and XprVector.\n"
	 << " * \\ingroup " << fun->group() << "\n"
	 << " */\n\n";
    }
    return os;
  }

  // unary functions
  template<class Stream>
  Stream& unary(Stream& os) const {
    FunctionBase::unary(os);

    for(ufun_iterator fun = ufun().begin(); fun != ufun().end(); ++fun) {
      os << "/**\n"
	 << " * \\fn " << fun->name() << "(const Vector<T, Sz>& rhs)\n"
	 << " * \\brief " << fun->description() << " function on Vector.\n"
	 << " * \\ingroup " << fun->group() << "\n"
	 << " */\n\n";
    }
    return os;
  }
};




class MatrixFunctions : public FunctionBase
{
public:
  MatrixFunctions() { }

public:
  template<class Stream>
  Stream& operator()(Stream& os) const {
    header(os);

    binary(os);
    binary2(os);
    binary3(os);
    unary(os);

    footer(os);

    return os;
  }

public:
  template<class Stream>
  Stream& header(Stream& os) const {
    FunctionBase::header(os);

    os << "//\n"
       << "// MatrixFunctions.h\n"
       << "//\n\n";
    return os;
  }

  // binary functions
  template<class Stream>
  Stream& binary(Stream& os) const {
    FunctionBase::binary(os);

    for(bfun_iterator fun = bfun().begin(); fun != bfun().end(); ++fun) {
      os << "/**\n"
	 << " * \\fn " << fun->name() << "(const Matrix<T1, Rows, Cols>& lhs, const Matrix<T2, Cols, Cols>& rhs)\n"
	 << " * \\brief " << fun->description() << " function for two Matrizes.\n"
	 << " * \\ingroup " << fun->group() << "\n"
	 << " */\n\n";
    }
    return os;
  }

  // binary functions with pod and std::complex<>
  template<class Stream>
  Stream& binary2(Stream& os) const {
    for(Type::const_iterator tp = m_type.begin(); tp != m_type.end(); ++tp) {
      for(bfun_iterator fun = bfun().begin(); fun != bfun().end(); ++fun) {
	os << "/**\n"
	   << " * \\fn " << fun->name() << "(const Matrix<T, Rows, Cols>& lhs, " << tp->name() << " rhs)\n"
	   << " * \\brief " << fun->description() << " function on Matrix and " << tp->description() << ".\n"
	   << " * \\ingroup " << fun->group() << "\n"
	   << " */\n\n";

// 	os << "/**\n"
// 	   << " * \\fn " << fun->name() << "(" << tp->name() << " lhs, const Matrix<T, Rows, Cols>& rhs)\n"
// 	   << " * \\brief " << fun->description() << " function on " << tp->description() << " and Matrix.\n"
// 	   << " * \\ingroup " << fun->group() << "\n"
// 	   << " */\n\n";
      }
    }
    return os;
  }

  // binary functions with expressions
  template<class Stream>
  Stream& binary3(Stream& os) const {
    for(bfun_iterator fun = bfun().begin(); fun != bfun().end(); ++fun) {
      os << "/**\n"
	 << " * \\fn " << fun->name() << "(const XprMatrix<E, Rows, Cols>& lhs, const Matrix<T, Rows, Cols>& rhs)\n"
	 << " * \\brief " << fun->description() << " function on XprMatrix and Matrix.\n"
	 << " * \\ingroup " << fun->group() << "\n"
	 << " */\n\n";

      os << "/**\n"
	 << " * \\fn " << fun->name() << "(const Matrix<T, Rows, Cols>& lhs, const XprMatrix<E, Rows, Cols>& rhs)\n"
	 << " * \\brief " << fun->description() << " function on Matrix and XprMatrix.\n"
	 << " * \\ingroup " << fun->group() << "\n"
	 << " */\n\n";
    }
    return os;
  }

  // unary functions
  template<class Stream>
  Stream& unary(Stream& os) const {
    FunctionBase::unary(os);

    for(ufun_iterator fun = ufun().begin(); fun != ufun().end(); ++fun) {
      os << "/**\n"
	 << " * \\fn " << fun->name() << "(const Matrix<T, Rows, Cols>& rhs)\n"
	 << " * \\brief " << fun->description() << " function on Matrix.\n"
	 << " * \\ingroup " << fun->group() << "\n"
	 << " */\n\n";
    }
    return os;
  }
};



int main()
{
  XprFunctions xpr_fun;
  VectorFunctions vec_fun;
  MatrixFunctions mtx_fun;

  Function::doxy_groups(std::cout);

  xpr_fun(std::cout);
  vec_fun(std::cout);
  mtx_fun(std::cout);
}
