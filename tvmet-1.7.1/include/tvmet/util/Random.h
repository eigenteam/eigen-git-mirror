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
 * $Id: Random.h,v 1.3 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_UTIL_RANDOM_H
#define TVMET_UTIL_RANDOM_H

#include <tvmet/CompileTimeError.h>

namespace tvmet {

namespace util {


/**
 * \class Random Random.h "tvmet/util/Random.h"
 * \brief A simple random class.
 * On each access this class returns a new random number using
 * std::rand(). The range generated is templated by MIN and
 * MAX.
 * \ingroup _util_function
 *
 * \par Example:
 * \code
 * #include <algorithm>
 *
 * tvmet::Random<int, 0, 100>				random;
 *
 * std::generate(m1.begin(), m1.end(), random());
 * \endcode
 */
template<class T, int MIN=0, int MAX=100>
class Random {
  static unsigned int				s_seed;
public:
  typedef T					value_type;
  Random() { TVMET_CT_CONDITION(MIN<MAX, wrong_random_range) }
  value_type operator()() {
    s_seed += (unsigned)std::time(0);
    std::srand(s_seed);
    return MIN + int(double(MAX) * std::rand()/(double(RAND_MAX)+1.0));
  }
};
// instance
template<class T, int MIN, int MAX>
unsigned int Random<T, MIN, MAX>::s_seed;


#if defined(TVMET_HAVE_COMPLEX)
/**
 * \class Random< std::complex<T> > Random.h "tvmet/util/Random.h"
 * \brief Specialized Random class.
 * \ingroup _util_function
 */
template<class T, int MIN=0, int MAX=100>
class Random {
  static unsigned int				s_seed;
public:
  typedef std::complex<T>			value_type;
  Random() { TVMET_CT_CONDITION(MIN<MAX, wrong_random_range) }
  value_type operator()() {
    s_seed += (unsigned)std::time(0);
    std::srand(s_seed);
    return MIN + int(double(MAX) * std::rand()/(double(RAND_MAX)+1.0));
  }
};
// instance
template<class T, int MIN, int MAX>
unsigned int Random<std::complex<T>, MIN, MAX>::s_seed;
#endif // defined(TVMET_HAVE_COMPLEX)


} // namespace util

} // namespace tvmet

#endif // TVMET_UTIL_RANDOM_H

// Local Variables:
// mode:C++
// End:
