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
 * $Id: TestUnFunc.cc,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#include <iostream>

#include <TestUnFunc.h>
#include <cppunit/extensions/HelperMacros.h>

/*****************************************************************************
 * Implementation Part I (cppunit integer part)
 ****************************************************************************/

/*
 * Is specialized for int, therefore it's like a normal class definition
 * and can placed into a cc file.
 */
TestUnFunc<int>::TestUnFunc()
  : vZero(0), vOne(1), vMinusOne(-1), vTwo(2)
  , mZero(0), mOne(1), mMinusOne(-1), mTwo(2)
{ }

void TestUnFunc<int>::setUp() { }

void TestUnFunc<int>::tearDown() { }


/*****************************************************************************
 * Implementation Part II (integer part)
 ****************************************************************************/
void TestUnFunc<int>::fn_abs() {
  vector_type v;
  matrix_type m;

  // abs
  v = abs(vMinusOne);
  m = abs(mMinusOne);
  CPPUNIT_ASSERT( all_elements(v == vOne) );
  CPPUNIT_ASSERT( all_elements(m == mOne) );
}


/****************************************************************************
 * instance
 ****************************************************************************/

CPPUNIT_TEST_SUITE_REGISTRATION( TestUnFunc<double> );
CPPUNIT_TEST_SUITE_REGISTRATION( TestUnFunc<int> );
