#include "clBLAS.h"

#include "ClBlasInstance.h"

#include <iostream>
using namespace std;

#define PUBLIC

PUBLIC ClBlasInstance::ClBlasInstance() {
    // cerr << "initializing clblas" << endl;
    clblasSetup();
}

PUBLIC ClBlasInstance::~ClBlasInstance() {
    // cerr << "clblas teardown" << endl;
    clblasTeardown();
}

//bool ClBlasInstance::initialized = false;

// assume single-threaded, at least for now
//void ClBlasInstance::initializeIfNecessary() {
//    if(!initialized) {
//        cerr << "initializing clblas" << endl;
//        clblasSetup();
//        initialized = true;
//    }
//}

