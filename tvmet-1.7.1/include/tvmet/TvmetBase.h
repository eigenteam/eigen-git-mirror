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
 * $Id: TvmetBase.h,v 1.11 2004/06/10 16:36:55 opetzold Exp $
 */

#ifndef TVMET_BASE_H
#define TVMET_BASE_H

#include <iosfwd>				// io streams forward declaration
#include <typeinfo>				// rtti: used by Xpr.h level printing
#include <cmath>				// unary and binary math
#include <cstdlib>				// labs

namespace tvmet {


/**
 * \class TvmetBase TvmetBase.h "tvmet/TvmetBase.h"
 * \brief Base class
 * Used for static polymorph call of print_xpr
 */
template<class E> class TvmetBase { };


/**
 * \class IndentLevel TvmetBase.h "tvmet/TvmetBase.h"
 * \brief Prints the level indent.
 */
class IndentLevel : public TvmetBase< IndentLevel >
{
public:
  IndentLevel(std::size_t level) : m_level(level) { }

  std::ostream& print_xpr(std::ostream& os) const {
    for(std::size_t i = 0; i != m_level; ++i) os << "   ";
    return os;
  }

private:
  std::size_t 					m_level;
};


/**
 * \fn operator<<(std::ostream& os, const TvmetBase<E>& e)
 * \brief overloaded ostream operator using static polymorphic.
 * \ingroup _binary_operator
 */
template<class E>
inline
std::ostream& operator<<(std::ostream& os, const TvmetBase<E>& e) {
  static_cast<const E&>(e).print_xpr(os);
  return os;
}


/**
 * \class dispatch TvmetBase.h "tvmet/TvmetBase.h"
 * \brief Class helper to distuingish between e.g. meta
 *        and loop strategy used.
 */
template<bool> struct dispatch;

/**
 * \class dispatch<true> TvmetBase.h "tvmet/TvmetBase.h"
 * \brief specialized.
 */
template<> struct dispatch<true>  { };

/**
 * \class dispatch<false> TvmetBase.h "tvmet/TvmetBase.h"
 * \brief specialized.
 */
template<> struct dispatch<false> { };


} // namespace tvmet

#endif // TVMET_BASE_H

// Local Variables:
// mode:C++
// End:
