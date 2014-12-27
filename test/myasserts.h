// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

//#include "MyException.h"

#include <stdexcept>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdlib>

template<typename T>
std::string toString(T val ) { // not terribly efficient, but works...
   std::ostringstream myostringstream;
   myostringstream << val;
   return myostringstream.str();
}
template<typename T>
void assertEquals( T one, T two ) {
    if( one != two ) {
        throw std::runtime_error( "assertEquals fail " + toString(one) + " != " + toString(two) );
    }
}

void assertEquals( float one, int two ) {
   assertEquals( one, (float)two );
}
void assertEquals( int one, float two ) {
   assertEquals( (float)one, two );
}

