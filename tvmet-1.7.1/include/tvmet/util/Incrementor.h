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
 * $Id: Incrementor.h,v 1.3 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_UTIL_INCREMENTOR_H
#define TVMET_UTIL_INCREMENTOR_H

namespace tvmet {

namespace util {


/**
 * \class Incrementor Incrementor.h "tvmet/util/Incrementor.h"
 * \brief A simple incrementor class.
 * The start value is given at construction time. After
 * each access the class increments the internal counter.
 * \ingroup _util_function
 *
 * \par Example:
 * \code
 * #include <algorithm>
 *
 * using namespace tvmet;
 *
 * ...
 *
 * std::generate(m1.begin(), m1.end(),
 * util::Incrementor<typename matrix_type::value_type>());
 * \endcode
 */
template<class T>
struct Incrementor
{
  Incrementor(T start=0) : m_inc(start) { }
  T operator()() { m_inc+=1; return m_inc; }

private:
  T 							m_inc;
};


#if defined(EIGEN_USE_COMPLEX)
/**
 * \class Incrementor< std::complex<T> > Incrementor.h "tvmet/util/Incrementor.h"
 * \brief Specialized Incrementor class.
 * \ingroup _util_function
 */
template<class T>
struct Incrementor< std::complex<T> > {
  Incrementor(const std::complex<T>& start=0)
    : m_inc(start) { }
  std::complex<T> operator()() {
    m_inc += std::complex<T>(1,1);
    return m_inc;
  }
private:
  std::complex<T>       				m_inc;
};
#endif // defined(EIGEN_USE_COMPLEX)


} // namespace util

} // namespace tvmet

#endif // TVMET_UTIL_INCREMENTOR_H

// Local Variables:
// mode:C++
// End:
