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
 * $Id: main.cc,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#include <iostream>

#include <cppunit/TextTestResult.h>
#include <cppunit/TestSuite.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>

using std::cout;
using std::endl;

int main()
{
  CppUnit::TextUi::TestRunner runner;

  // retreive the instance of the TestFactoryRegistry
  CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();

  // obtain and add a new TestSuite created by the TestFactoryRegistry
  runner.addTest( registry.makeTest() );

  // Run the test.
  bool wasSucessful = runner.run();

  // Return error code 1 if the one of test failed; disturbs
  // the distrib process due to the return error code -> no return code
  return wasSucessful ? 0 : 1;
}
