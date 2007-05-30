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
 * $Id: AliasProxy.h,v 1.4 2004/06/10 16:36:55 opetzold Exp $
 */

#ifndef TVMET_ALIAS_PROXY_H
#define TVMET_ALIAS_PROXY_H

namespace tvmet {


/** forwards */
template<class E> class AliasProxy;


/**
 * \brief Simplify syntax for alias Matrices and Vectors,
 *        where aliasing left hand values appear in the
 *        expression.
 * \par Example:
 * \code
 * typedef tvmet::Matrix<double, 10, 10>	matrix_type;
 * matrix_type					m;
 * ...
 * alias(m) += trans(m);
 * \endcode
 * \sa AliasProxy
 * \sa Some Notes \ref alias
 */
template<class E>
AliasProxy<E> alias(E& expr) { return AliasProxy<E>(expr); }


/**
 * \class AliasProxy AliasProxy.h "tvmet/AliasProxy.h"
 * \brief Assign proxy for alias Matrices and Vectors.
 *
 *        A short lived object to provide simplified alias syntax.
 *        Only the friend function alias is allowed to create
 *        such a object. The proxy calls the appropriate member
 *        alias_xyz() which have to use temporaries to avoid
 *        overlapping memory regions.
 * \sa alias
 * \sa Some Notes \ref alias
 * \note Thanks to ublas-dev group, where the principle idea
 *       comes from.
 */
template<class E>
class AliasProxy
{
  AliasProxy(const AliasProxy&);
  AliasProxy& operator=(const AliasProxy&);

  friend AliasProxy<E> alias<>(E& expr);

public:
  AliasProxy(E& expr) : m_expr(expr) { }


  template<class E2>
  E& operator=(const E2& expr) {
    return m_expr.alias_assign(expr);
  }

  template<class E2>
  E& operator+=(const E2& expr) {
    return m_expr.alias_add_eq(expr);
  }

  template<class E2>
  E& operator-=(const E2& expr) {
    return m_expr.alias_sub_eq(expr);
  }

  template<class E2>
  E& operator*=(const E2& expr) {
    return m_expr.alias_mul_eq(expr);
  }

  template<class E2>
  E& operator/=(const E2& expr) {
    return m_expr.alias_div_eq(expr);
  }

private:
  E&						m_expr;
};


#if 0
namespace element_wise {
// \todo to write
template<class E, class E2>
E& operator/=(AliasProxy<E>& proxy, const E2& rhs) {
  return proxy.div_upd(rhs);
}

}
#endif


} // namespace tvmet


#endif /* TVMET_ALIAS_PROXY_H */

// Local Variables:
// mode:C++
// End:
