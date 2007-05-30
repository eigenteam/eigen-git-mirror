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
 * $Id: dox_operators.cc,v 1.6 2003/11/30 08:26:25 opetzold Exp $
 */

#include <iostream>
#include <vector>
#include <string>

#include <tvmet/config.h>

#include "Util.h"

class OperatorBase
{
public:
  OperatorBase() {
    m_binary_operators.push_back( BinaryOperator("+", "Addition") );
    m_binary_operators.push_back( BinaryOperator("-", "Subtraction") );
    m_binary_operators.push_back( BinaryOperator("*", "Multliply") );
    m_binary_operators.push_back( BinaryOperator("/", "Division") );
    m_binary_operators.push_back( BinaryOperator("%", "Modulo", true) );
    m_binary_operators.push_back( BinaryOperator("^", "Exclusive OR", true) );
    m_binary_operators.push_back( BinaryOperator("&", "AND", true) );
    m_binary_operators.push_back( BinaryOperator("|", "OR", true) );
    m_binary_operators.push_back( BinaryOperator("<<", "Left Shift", true) );
    m_binary_operators.push_back( BinaryOperator(">>", "Right Shift", true) );
    m_binary_operators.push_back( BinaryOperator(">", "Bigger") );
    m_binary_operators.push_back( BinaryOperator("<", "Lesser") );
    m_binary_operators.push_back( BinaryOperator(">=", "Bigger Equal") );
    m_binary_operators.push_back( BinaryOperator("<=", "Less Equal") );
    m_binary_operators.push_back( BinaryOperator("==", "Equal") );
    m_binary_operators.push_back( BinaryOperator("!=", "Not Equal") );
    m_binary_operators.push_back( BinaryOperator("&&", "Logical AND", true) );
    m_binary_operators.push_back( BinaryOperator("||", "Logical OR", true) );

    m_unary_operators.push_back( UnaryOperator("!", "Logical Not", true) );
    m_unary_operators.push_back( UnaryOperator("~", "Bitwise Not", true) );
    m_unary_operators.push_back( UnaryOperator("-", "Negate") );
  }

  virtual ~OperatorBase() { }

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
  typedef std::vector< BinaryOperator >::const_iterator	bop_iterator;
  typedef std::vector< UnaryOperator >::const_iterator	uop_iterator;

public:
  virtual const std::vector< BinaryOperator >& bop() const { return m_binary_operators; }
  virtual const std::vector< UnaryOperator >& uop() const { return m_unary_operators; }

protected:
  std::vector< BinaryOperator >		m_binary_operators;
  std::vector< UnaryOperator >		m_unary_operators;
  Type					m_type;
};


class XprOperators : public OperatorBase
{
public:
  XprOperators() { }

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
    OperatorBase::header(os);

    os << "//\n"
       << "// XprOperators.h\n"
       << "//\n\n";
    return os;
  }

  // global binary math, bitops and logical operators
  template<class Stream>
  Stream& binary(Stream& os) const {
    OperatorBase::binary(os);

    for(bop_iterator op = bop().begin(); op != bop().end(); ++op) {
      os << "/**\n"
	 << " * \\fn operator" << op->symbol() << "(const XprVector<E1, Sz>& lhs, const XprVector<E2, Sz>& rhs)\n"
	 << " * \\brief " << op->description() << " operator for two XprVector\n"
	 << " * \\ingroup " << op->group() << "\n"
	 << " */\n\n";

      os << "/**\n"
	 << " * \\fn operator" << op->symbol() << "(const XprMatrix<E1, Rows1, Cols1>& lhs, const XprMatrix<E2, Cols1, Cols2>& rhs)\n"
	 << " * \\brief " << op->description() << " operator for two XprMatrix.\n"
	 << " * \\ingroup " << op->group() << "\n"
	 << " */\n\n";
    }
    return os;
  }

  // global binary math, bitops and logical operators with pod and std::complex<>
  template<class Stream>
  Stream& binary2(Stream& os) const {
    for(Type::const_iterator tp = m_type.begin(); tp != m_type.end(); ++tp) {
      for(bop_iterator op = bop().begin(); op != bop().end(); ++op) {
	if(tp->is_int() && op->int_only()) {
	  os << "/**\n"
	     << " * \\fn operator" << op->symbol() << "(const XprVector<E, Sz>& lhs, " << tp->name() << " rhs)\n"
	     << " * \\brief " << op->description() << " operator between XprVector and " << tp->description() << ".\n"
	     << " * \\ingroup " << op->group() << "\n"
	     << " */\n\n";

	  os << "/**\n"
	     << " * \\fn operator" << op->symbol() << "(" << tp->name() << " lhs, const XprVector<E, Sz>& rhs)\n"
	     << " * \\brief " << op->description() << " operator between " << tp->description() << " and XprVector.\n"
	     << " * \\ingroup " << op->group() << "\n"
	     << " */\n\n";

	  os << "/**\n"
	     << " * \\fn operator" << op->symbol() << "(const XprMatrix<E, Rows, Cols>& lhs, " << tp->name() << " rhs)\n"
	     << " * \\brief " << op->description() << " operator between XprMatrix and " << tp->description() << ".\n"
	     << " * \\ingroup " << op->group() << "\n"
	     << " */\n\n";

	  os << "/**\n"
	     << " * \\fn operator" << op->symbol() << "(" << tp->name() << " lhs, const XprMatrix<E, Rows, Cols>& rhs)\n"
	     << " * \\brief " << op->description() << " operator between " << tp->description() << " and XprMatrix.\n"
	     << " * \\ingroup " << op->group() << "\n"
	     << " */\n\n";
	}
      }
    }
    return os;
  }

  // global unary operators
  template<class Stream>
  Stream& unary(Stream& os) const {
    OperatorBase::unary(os);

    for(uop_iterator op = uop().begin(); op != uop().end(); ++op) {
      os << "/**\n"
	 << " * \\fn operator" << op->symbol() << "(const XprVector<E, Sz>& rhs)\n"
	 << " * \\brief " << op->description() << " operator for XprVector\n"
	 << " * \\ingroup " << op->group() << "\n"
	 << " */\n\n";

      os << "/**\n"
	 << " * \\fn operator" << op->symbol() << "(const XprMatrix<E, Rows, Cols>& rhs)\n"
	 << " * \\brief " << op->description() << " operator for XprMatrix.\n"
	 << " * \\ingroup " << op->group() << "\n"
	 << " */\n\n";
    }
    return os;
  }
};


class VectorOperators : public OperatorBase
{
public:
  VectorOperators() { }

public:
  template<class Stream>
  Stream& operator()(Stream& os) const {
    header(os);

    binary(os);
    binary2(os);
    binary3(os);
    unary(os);

    footer(os);
  }

public:
  template<class Stream>
  Stream& header(Stream& os) const {
    OperatorBase::header(os);

    os << "//\n"
       << "// VectorOperators.h\n"
       << "//\n\n";
    return os;
  }

  // global binary math, bitops and logical operators
  template<class Stream>
  Stream& binary(Stream& os) const {
    OperatorBase::binary(os);

    for(bop_iterator op = bop().begin(); op != bop().end(); ++op) {
      os << "/**\n"
	 << " * \\fn operator" << op->symbol() << "(const Vector<T1, Sz>& lhs, const Vector<T2, Sz>& rhs)\n"
	 << " * \\brief " << op->description() << " operator for two Vector\n"
	 << " * \\ingroup " << op->group() << "\n"
	 << " */\n\n";
    }
    return os;
  }

  // global binary math, bitops and logical operators with pod and std::complex<>
  template<class Stream>
  Stream& binary2(Stream& os) const {
    for(Type::const_iterator tp = m_type.begin(); tp != m_type.end(); ++tp) {
      for(bop_iterator op = bop().begin(); op != bop().end(); ++op) {
	if(tp->is_int() && op->int_only()) {
	  os << "/**\n"
	     << " * \\fn operator" << op->symbol() << "(const Vector<T, Sz>& lhs, " << tp->name() << " rhs)\n"
	     << " * \\brief " << op->description() << " operator between Vector and " << tp->description() << ".\n"
	     << " * \\ingroup _operators\n"
	     << " * \\ingroup " << op->group() << "\n";

	  os << "/**\n"
	     << " * \\fn operator" << op->symbol() << "(" << tp->name() << " lhs, const Vector<T, Sz>& rhs)\n"
	     << " * \\brief " << op->description() << " operator between " << tp->description() << " and Vector.\n"
	     << " * \\ingroup " << op->group() << "\n"
	     << " */\n\n";
	}
      }
    }
    return os;
  }

  // global binary operations with expressions
  template<class Stream>
  Stream& binary3(Stream& os) const {
    for(bop_iterator op = bop().begin(); op != bop().end(); ++op) {
      os << "/**\n"
	 << " * \\fn operator" << op->symbol() << "(const XprVector<E, Sz>& lhs, const Vector<T, Sz>& rhs)\n"
	 << " * \\brief " << op->description() << " operator between XprVector and Vector\n"
	 << " * \\ingroup " << op->group() << "\n"
	 << " */\n\n";

      os << "/**\n"
	 << " * \\fn operator" << op->symbol() << "(const Vector<T, Sz>& lhs, const XprVector<E, Sz>& rhs)\n"
	 << " * \\brief " << op->description() << " operator between Vector and XprVector\n"
	 << " * \\ingroup " << op->group() << "\n"
	 << " */\n\n";
    }
    return os;
  }

  // global unary operators
  template<class Stream>
  Stream& unary(Stream& os) const {
    OperatorBase::unary(os);

    for(uop_iterator op = uop().begin(); op != uop().end(); ++op) {
      os << "/**\n"
	 << " * \\fn operator" << op->symbol() << "(const Vector<T, Sz>& rhs)\n"
	 << " * \\brief " << op->description() << " operator for Vector\n"
	 << " * \\ingroup " << op->group() << "\n"
	 << " */\n\n";
    }
    return os;
  }
};



class MatrixOperators : public OperatorBase
{
public:
  MatrixOperators() { }

public:
  template<class Stream>
  Stream& operator()(Stream& os) const {
    header(os);

    binary(os);
    binary2(os);
    binary3(os);
    unary(os);

    footer(os);
  }

public:
  template<class Stream>
  Stream& header(Stream& os) const {
    OperatorBase::header(os);

    os << "//\n"
       << "// MatrixOperators.h\n"
       << "//\n\n";
    return os;
  }

  // global binary math, bitops and logical operators
  template<class Stream>
  Stream& binary(Stream& os) const {
    OperatorBase::binary(os);

    for(bop_iterator op = bop().begin(); op != bop().end(); ++op) {
      os << "/**\n"
	 << " * \\fn operator" << op->symbol() << "(const Matrix<T1, Rows1, Cols1>& lhs, const Matrix<T2, Cols1, Cols2>& rhs)\n"
	 << " * \\brief " << op->description() << " operator for two Matrizes\n"
	 << " * \\ingroup " << op->group() << "\n"
	 << " */\n\n";
    }
    return os;
  }

  // global binary math, bitops and logical operators with pod and std::complex<>
  template<class Stream>
  Stream& binary2(Stream& os) const {
    for(Type::const_iterator tp = m_type.begin(); tp != m_type.end(); ++tp) {
      for(bop_iterator op = bop().begin(); op != bop().end(); ++op) {
	if(tp->is_int() && op->int_only()) {
	  os << "/**\n"
	     << " * \\fn operator" << op->symbol() << "(const Matrix<T, Rows, Cols>& lhs, " << tp->name() << " rhs)\n"
	     << " * \\brief " << op->description() << " operator between Vector and " << tp->description() << ".\n"
	     << " * \\ingroup " << op->group() << "\n"
	     << " */\n\n";

	  os << "/**\n"
	     << " * \\fn operator" << op->symbol() << "(" << tp->name() << " lhs, const Matrix<T, Rows, Cols>& rhs)\n"
	     << " * \\brief " << op->description() << " operator between " << tp->description() << " and Vector.\n"
	     << " * \\ingroup " << op->group() << "\n"
	     << " */\n\n";
	}
      }
    }
    return os;
  }

  // global binary operations with expressions
  template<class Stream>
  Stream& binary3(Stream& os) const {
    for(bop_iterator op = bop().begin(); op != bop().end(); ++op) {
      os << "/**\n"
	 << " * \\fn operator" << op->symbol() << "(const XprMatrix<E, Rows, Cols>& lhs, const Matrix<T, Rows, Cols>& rhs)\n"
	 << " * \\brief " << op->description() << " operator between XprMatrix and Matrix\n"
	 << " * \\ingroup " << op->group() << "\n"
	 << " */\n\n";

      os << "/**\n"
	 << " * \\fn operator" << op->symbol() << "(const Matrix<T, Rows, Cols>& lhs, const XprMatrix<E, Rows, Cols>& rhs)\n"
	 << " * \\brief " << op->description() << " operator between Matrix and XprMatrix\n"
	 << " * \\ingroup " << op->group() << "\n"
	 << " */\n\n";
    }
    return os;
  }

  // global unary operators
  template<class Stream>
  Stream& unary(Stream& os) const {
    OperatorBase::unary(os);

    for(uop_iterator op = uop().begin(); op != uop().end(); ++op) {
      os << "/**\n"
	 << " * \\fn operator" << op->symbol() << "(const Matrix<T, Rows, Cols>& rhs)\n"
	 << " * \\brief " << op->description() << " operator for Matrix\n"
	 << " * \\ingroup " << op->group() << "\n"
	 << " */\n\n";
    }
    return os;
  }
};

int main()
{
  XprOperators xpr_op;
  VectorOperators vec_op;
  MatrixOperators mtx_op;

  Operator::doxy_groups(std::cout);

  xpr_op(std::cout);
  //vec_op(std::cout);
  //mtx_op(std::cout);
}
