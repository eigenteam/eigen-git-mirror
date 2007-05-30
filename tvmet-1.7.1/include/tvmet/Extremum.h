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
 * $Id: Extremum.h,v 1.6 2003/11/30 08:26:25 opetzold Exp $
 */

#ifndef TVMET_EXTREMUM_H
#define TVMET_EXTREMUM_H

namespace tvmet {


/**
 * \class matrix_tag Extremum.h "tvmet/Extremum.h"
 * \brief For use with Extremum to simplify max handling.
 * This allows the min/max functions to return an Extremum object.
 */
struct matrix_tag { };


/**
 * \class vector_tag Extremum.h "tvmet/Extremum.h"
 * \brief For use with Extremum to simplify max handling.
 * This allows the min/max functions to return an Extremum object.
 */
struct vector_tag { };


/**
 * \class Extremum Extremum.h "tvmet/Extremum.h"
 * \brief Generell class for storing extremums determined by min/max.
 */
template<class T1, class T2, class Tag>
class Extremum { };


/**
 * \class Extremum<T1, T2, vector_tag> Extremum.h "tvmet/Extremum.h"
 * \brief Partial specialzed for vectors to store extremums by value and index.
 */
template<class T1, class T2>
class Extremum<T1, T2, vector_tag>
{
public:
  typedef T1					value_type;
  typedef T2					index_type;

public:
  Extremum(value_type value, index_type index)
    : m_value(value), m_index(index) { }
  value_type value() const { return m_value; }
  index_type index() const { return m_index; }

private:
  value_type 					m_value;
  index_type 					m_index;
};


/**
 * \class Extremum<T1, T2, matrix_tag> Extremum.h "tvmet/Extremum.h"
 * \brief Partial specialzed for matrix to store extremums by value, row and column.
 */
template<class T1, class T2>
class Extremum<T1, T2, matrix_tag>
{
public:
  typedef T1					value_type;
  typedef T2					index_type;

public:
  Extremum(value_type value, index_type row, index_type col)
    : m_value(value), m_row(row), m_col(col) { }
  value_type value() const { return m_value; }
  index_type row() const { return m_row; }
  index_type col() const { return m_col; }

private:
  value_type 					m_value;
  index_type 					m_row, m_col;
};


} // namespace tvmet

#endif // TVMET_EXTREMUM_H

// Local Variables:
// mode:C++
// End:
