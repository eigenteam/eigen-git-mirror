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
 * $Id: TestNumericTraits.cc,v 1.1 2004/09/15 07:51:43 opetzold Exp $
 */

#include <iostream>
#include <complex>

#include <TestNumericTraits.h>
#include <cppunit/extensions/HelperMacros.h>


/****************************************************************************
 * instance
 ****************************************************************************/

CPPUNIT_TEST_SUITE_REGISTRATION( TestNumericTraits<char> );
CPPUNIT_TEST_SUITE_REGISTRATION( TestNumericTraits<unsigned char> );
CPPUNIT_TEST_SUITE_REGISTRATION( TestNumericTraits<short int> );
CPPUNIT_TEST_SUITE_REGISTRATION( TestNumericTraits<unsigned short int> );
CPPUNIT_TEST_SUITE_REGISTRATION( TestNumericTraits<int> );
CPPUNIT_TEST_SUITE_REGISTRATION( TestNumericTraits<unsigned int> );
CPPUNIT_TEST_SUITE_REGISTRATION( TestNumericTraits<long> );
CPPUNIT_TEST_SUITE_REGISTRATION( TestNumericTraits<unsigned long> );

CPPUNIT_TEST_SUITE_REGISTRATION( TestNumericTraits<float> );
CPPUNIT_TEST_SUITE_REGISTRATION( TestNumericTraits<double> );
#if defined(TVMET_HAVE_LONG_DOUBLE)
CPPUNIT_TEST_SUITE_REGISTRATION( TestNumericTraits<long double> );
#endif
