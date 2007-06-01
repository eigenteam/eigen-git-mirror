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
 * $Id: Literal.h,v 1.9 2003/11/30 18:35:17 opetzold Exp $
 */

#ifndef TVMET_XPR_LITERAL_H
#define TVMET_XPR_LITERAL_H

namespace tvmet {


/**
 * \class XprLiteral Literal.h "tvmet/xpr/Literal.h"
 * \brief Specify literals like scalars into the expression.
 *        This expression is used for vectors and matrices - the
 *        decision is done by the access operator.
 */
template<class T>
class XprLiteral
  : public TvmetBase< XprLiteral<T> >
{
  XprLiteral();
  XprLiteral& operator=(const XprLiteral&);

public:
  typedef T						value_type;

public:
  /** Complexity counter. */
  enum {
    ops       = 1
  };

public:
  /** Constructor by value for literals . */
  explicit XprLiteral(value_type value)
    : m_data(value)
  { }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprLiteral(const XprLiteral& e)
    : m_data(e.m_data)
  { }
#endif

  /** Index operator, gives the value for vectors. */
  value_type operator()(int) const { return m_data; }

  /** Index operator for arrays/matrices. */
  value_type operator()(int, int) const { return m_data; }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, int l=0) const {
    os << IndentLevel(l++) << "XprLiteral[O=" << ops << "]<T="
       << typeid(value_type).name()
       << ">," << std::endl;
  }

private:
  const value_type 					m_data;
};


} // namespace tvmet

#endif // TVMET_XPR_LITERAL_H

// Local Variables:
// mode:C++
// End:
