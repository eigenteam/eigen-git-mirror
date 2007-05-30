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
 * $Id: Util.h,v 1.5 2003/12/19 18:01:37 opetzold Exp $
 */

#ifndef TVMET_DOC_UTIL_H
#define TVMET_DOC_UTIL_H

#include <vector>
#include <tvmet/config.h>

struct Function {
  Function() { }
  virtual ~Function() { }
  virtual const char* name() const = 0;
  virtual const char* description() const = 0;
  virtual bool int_only() const = 0;
  static const char* group() { return "_function"; }
  static const char* group_unary() { return "_unary_function"; }
  static const char* group_binary() { return "_binary_function"; }
  template<class Stream> static Stream& doxy_groups(Stream& os) {
    os << "/**\n"
       << " * \\defgroup " << group() << " Global Functions\n"
       << " */\n\n";
    os << "/**\n"
       << " * \\defgroup " << group_unary() << " Global Unary Functions\n"
       << " * \\ingroup " << group() << "\n"
       << " */\n\n";
    os << "/**\n"
       << " * \\defgroup " << group_binary() << " Global Binary Functions\n"
       << " * \\ingroup " << group() << "\n"
       << " */\n\n";
    os << "/**\n"
       << " * \\defgroup " << "_trinary_function" << " Global Trinary Functions\n"
       << " * \\ingroup " << group() << "\n"
       << " */\n\n";
    return os;
  }

};

class BinaryFunction : public Function {
public:
  BinaryFunction(const char* s, const char* d, bool i = false)
    : m_name(s), m_description(d), m_int_only(i) { }
  const char* name() const { return m_name; }
  const char* description() const { return m_description; }
  const char* group() const { return group_binary(); }
  bool int_only() const { return m_int_only; }
private:
  const char* 						m_name;
  const char* 						m_description;
  bool 							m_int_only;
};

class UnaryFunction : public Function {
public:
  UnaryFunction(const char* s, const char* d, bool i = false)
    : m_name(s), m_description(d), m_int_only(i)  { }
  virtual ~UnaryFunction() { }
  const char* name() const { return m_name; }
  const char* description() const { return m_description; }
  const char* group() const { return group_unary(); }
  bool int_only() const { return m_int_only; }
private:
  const char* 						m_name;
  const char* 						m_description;
  bool 							m_int_only;
};

struct Operator {
  Operator() { }
  virtual ~Operator() { }
  virtual const char* symbol() const = 0;
  virtual const char* description() const = 0;
  virtual bool int_only() const = 0;
  static const char* group() { return "_operator"; }
  static const char* group_unary() { return "_unary_operator"; }
  static const char* group_binary() { return "_binary_operator"; }
  template<class Stream> static Stream& doxy_groups(Stream& os) {
    os << "/**\n"
       << " * \\defgroup " << group() << " Global Operators\n"
       << " */\n\n";
    os << "/**\n"
       << " * \\defgroup " << group_binary() << " Global Binary Operators\n"
       << " * \\ingroup " << group() << "\n"
       << " */\n\n";
    os << "/**\n"
       << " * \\defgroup " << group_unary() << " Global Unary Operators\n"
       << " * \\ingroup " << group() << "\n"
       << " */\n\n";
    return os;
  }
};

class BinaryOperator : public Operator {
public:
  BinaryOperator(const char* s, const char* d, bool i = false)
    : m_symbol(s), m_description(d), m_int_only(i) { }
  virtual ~BinaryOperator() { }
  const char* symbol() const { return m_symbol; }
  const char* description() const { return m_description; }
  const char* group() const { return group_binary(); }
 bool int_only() const { return m_int_only; }
private:
  const char* 						m_symbol;
  const char* 						m_description;
  bool 							m_int_only;
};

class UnaryOperator : public Operator {
public:
  UnaryOperator(const char* s, const char* d, bool i = false)
    : m_symbol(s), m_description(d), m_int_only(i)  { }
  virtual ~UnaryOperator() { }
  const char* symbol() const { return m_symbol; }
  const char* description() const { return m_description; }
  const char* group() const { return group_unary(); }
  bool int_only() const { return m_int_only; }
private:
  const char* 						m_symbol;
  const char* 						m_description;
  bool 							m_int_only;
};

class DataType {
public:
  DataType(const char* s, const char* d, bool i = false)
    : m_name(s), m_description(d), m_is_int(i){ }
  const char* name() const { return m_name; }
  const char* description() const { return m_description; }
  bool is_int() const { return m_is_int; }
private:
  const char* 						m_name;
  const char* 						m_description;
  bool 							m_is_int;
};

class Type
{
public:
  Type() {
    datatypes.push_back( DataType("int", "int", true) );
    datatypes.push_back( DataType("float", "float") );
    datatypes.push_back( DataType("double", "double") );
#ifdef TVMET_HAVE_LONG_DOUBLE
    datatypes.push_back( DataType("long double", "long double") );
#endif // HAVE_LONG_DOUBLE
#ifdef TVMET_HAVE_COMPLEX
    datatypes.push_back( DataType("const std::complex<T>&", "std::complex<T>") );
#endif // HAVE_COMPLEX
  }

  virtual ~Type() { }

public:
  template<class Stream>
  Stream& header(Stream& os) const {
    os << "namespace tvmet {\n\n";
    return os;
  }

  template<class Stream>
  Stream& footer(Stream& os) const {
    os << "\n} // namespace tvmet\n\n";
    return os;
  }

public:
  typedef std::vector< DataType >::const_iterator	const_iterator;

public:
  const_iterator begin() const { return datatypes.begin(); }
  const_iterator end() const { return datatypes.end(); }

private:
  std::vector< DataType >				datatypes;
};

#endif // TVMET_DOC_UTIL_H

// Local Variables:
// mode:C++
// End:
