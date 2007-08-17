/* This file is part of Eigen, a C++ template library for linear algebra
 * Copyright (C) 2007 Benoit Jacob <jacob@math.jussieu.fr>
 *
 * Based on Tvmet source code, http://tvmet.sourceforge.net,
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
 * $Id: CommaInitializer.h,v 1.14 2005/03/02 12:14:22 opetzold Exp $
 */

#ifndef TVMET_COMMA_INITIALIZER_H
#define TVMET_COMMA_INITIALIZER_H

#include <tvmet/CompileTimeError.h>

namespace tvmet {

/**
 * \class CommaInitializer CommaInitializer.h "tvmet/CommaInitializer.h"
 * \brief Initialize classes using a comma separated lists.
 *
 * The comma operator is called when it appears next to an object of
 * the type the comma is defined for. However, "operator," is not called
 * for function argument lists, only for objects that are out in the open,
 * separated by commas (Thinking C++
 * <a href=http://www.ida.liu.se/~TDDA14/online/v1ticpp/Chapter12.html>
 * Ch.12: Operator comma</a>).
 *
 * This implementation uses the same technique as described in Todd Veldhuizen
 * Techniques for Scientific C++
 * <a href=http://extreme.indiana.edu/~tveldhui/papers/techniques/techniques01.html#l43>
 * chapter 1.11 Comma overloading</a>.
 *
 * The initializer list is avaible after instanciation of the object,
 * therefore use it like:
 * \code
 * vector3d t;
 * t = 1.0, 2.0, 3.0;
 * \endcode
 * It's evaluated to (((t = 1.0), 2.0), 3.0)
 *
 * For matrizes the initilization is done row wise.
 *
 * If the comma separted list of values longer then the size of the vector
 * or matrix a compile time error will occour. Otherwise the pending values
 * will be written random into the memory.
 *
 */
template<typename Obj, int LEN>
class CommaInitializer
{
  typedef typename Obj::value_type value_type;

  /**
   * \class Initializer
   * \brief Helper fo recursive overloaded comma operator.
   */
  template<int N> class Initializer
  {
    Initializer();
    Initializer& operator=(const Initializer&);

  public:
    Initializer(Obj& obj, int index) : m_obj(obj), m_index(index) {}

    /** Overloads the comma operator for recursive assign values from comma
	separated list. */
    Initializer<N+1> operator,(value_type rhs)
    {
      TVMET_CT_CONDITION(N < LEN, CommaInitializerList_is_too_long)
      m_obj.commaWrite(m_index, rhs);
      return Initializer<N+1>(m_obj, m_index+1);
    }

  private:
    Obj& m_obj;
    int m_index;
  };

public:
  CommaInitializer(const CommaInitializer& rhs)
    : m_object(rhs.m_object),
      m_data(rhs.m_data)
  {}

  /** Constructor used by Vector or Matrix operator(value_type rhs) */
  CommaInitializer(Obj& obj, value_type x)
    : m_object(obj),
      m_data(x)
  {}

  /** Destructor, does nothing. */
  ~CommaInitializer() {}

  /** Overloaded comma operator, called only once for the first occoured comma. This
      means the first value is assigned by %operator=() and the 2nd value after the
      comma. Therefore we call the %Initializer::operator,() for the list starting
      after the 2nd. */
  Initializer<2> operator,(value_type rhs)
  {
    m_object.commaWrite(0, m_data);
    m_object.commaWrite(1, rhs);
    return Initializer<2>(m_object, 2);
  }

private:
  Obj& m_object;
  value_type m_data;
};

} // namespace tvmet

#endif //  TVMET_COMMA_INITIALIZER_H

// Local Variables:
// mode:C++
// End:
