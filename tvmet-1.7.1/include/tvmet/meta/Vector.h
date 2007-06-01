/*
 * Tiny Vector Matrix Library
 * Dense Vector Matrix Libary of Tiny size using Expression Templates
 *
 * Copyright (C) 2001 - 2003 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * lesser General Public License for more details.
 *
 * You should have received a copy of the GNU lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: Vector.h,v 1.20 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_META_VECTOR_H
#define TVMET_META_VECTOR_H

#include <tvmet/NumericTraits.h>
#include <tvmet/xpr/Null.h>

namespace tvmet {

/* forwards */
template<class T, int Sz> class Vector;


namespace meta {


/**
 * \class Vector Vector.h "tvmet/meta/Vector.h"
 * \brief Meta %Vector class using expression templates
 */
template<int Sz, int K=0>
class Vector
{
  Vector();
  Vector(const Vector&);
  Vector& operator=(const Vector&);

private:
  enum {
    doIt = (K < (Sz-1)) ? 1 : 0		/**< recursive counter */
  };

public:
  /** assign an expression expr using the functional assign_fn. */
  template <class Dest, class Src, class Assign>
  static inline
  void assign(Dest& lhs, const Src& rhs, const Assign& assign_fn) {
    assign_fn.apply_on(lhs(K), rhs(K));
    meta::Vector<Sz * doIt, (K+1) * doIt>::assign(lhs, rhs, assign_fn);
  }

  /** build the sum of the vector. */
  template<class E>
  static inline
  typename E::value_type
  sum(const E& e) {
    return e(K) + meta::Vector<Sz * doIt, (K+1) * doIt>::sum(e);
  }

  /** build the product of the vector. */
  template<class E>
  static inline
  typename NumericTraits<
    typename E::value_type
  >::sum_type
  product(const E& e) {
    return e(K) * meta::Vector<Sz * doIt, (K+1) * doIt>::product(e);
  }

  /** build the dot product of the vector. */
  template<class Dest, class Src>
  static inline
  typename PromoteTraits<
    typename Dest::value_type,
    typename Src::value_type
  >::value_type
  dot(const Dest& lhs, const Src& rhs) {
    return lhs(K) * rhs(K)
      + meta::Vector<Sz * doIt, (K+1) * doIt>::dot(lhs, rhs);
  }

  /** check for all elements */
  template<class E>
  static inline
  bool
  all_elements(const E& e) {
    if(!e(K)) return false;
    return meta::Vector<Sz * doIt, (K+1) * doIt>::all_elements(e);
  }

  /** check for any elements */
  template<class E>
  static inline
  bool
  any_elements(const E& e) {
    if(e(K)) return true;
    return meta::Vector<Sz * doIt, (K+1) * doIt>::any_elements(e);
  }
};


/**
 * \class Vector<0,0> Vector.h "tvmet/meta/Vector.h"
 * \brief Meta %Vector Specialized for recursion
 */
template<>
class Vector<0,0>
{
  Vector();
  Vector(const Vector&);
  Vector& operator=(const Vector&);

public:
  template <class Dest, class Src, class Assign>
  static inline void assign(Dest&, const Src&, const Assign&) { }

  template<class E>
  static inline XprNull sum(const E&) { return XprNull(); }

  template<class E>
  static inline XprNull product(const E&) { return XprNull(); }

  template<class Dest, class Src>
  static inline XprNull dot(const Dest&, const Src&) { return XprNull(); }

  template<class E>
  static inline bool all_elements(const E&) { return true; }

  template<class E>
  static inline bool any_elements(const E&) { return false; }
};


} // namespace meta

} // namespace tvmet

#endif /* TVMET_META_VECTOR_H */

// Local Variables:
// mode:C++
// End:
