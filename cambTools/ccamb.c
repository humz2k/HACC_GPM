#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include <stdlib.h>
#include <math.h>

#include "cambpy.h"

#define getIndent(calls) char indent[50] = ""; for (int i = 0; i < (calls*6); i++){strcat(indent," ");}
//#include "ccamb.h"

#define VerboseCCamb

PyObject* pName;
PyObject* pModule;

int FIND_ME(){
    return 1;
}

void init_python(int calls){
    getIndent(calls);

    #ifdef VerboseCCamb
    printf("%sInitializing Python...\n",indent);
    #endif

    Py_Initialize();

    #ifdef VerboseCCamb
    printf("%s   Initialized Python.\n",indent);
    #endif

    //PyRun_SimpleString("import sys");
    //char tmp[400];
    //sprintf(tmp,"sys.path.append(\"%s\")",cambToolsPath);
    //printf("%s   cambToolsPath %s\n",indent,cambToolsPath);
    //PyRun_SimpleString(tmp);
    char cambpystr[] = CAMBPYSTR;
    PyRun_SimpleString(cambpystr);
    // Build the name object
    pName = PyUnicode_FromString((char*)"cambpymodule");

    #ifdef VerboseCCamb
    printf("%sImporting Numpy...\n",indent);
    #endif

    import_array();

    #ifdef VerboseCCamb
    printf("%s   Imported Numpy.\n",indent);
    // Load the module object

    printf("%sImporting Camb...\n",indent);
    #endif
    
    pModule = PyImport_Import(pName);
    
    #ifdef VerboseCCamb
    printf("%s   Imported Camb.\n",indent);
    #endif
}

void finalize_python(int calls){
    getIndent(calls);
    #ifdef VerboseCCamb
    printf("%sFinalizing Python...\n",indent);
    #endif

    //Py_DECREF(pModule);
    //Py_DECREF(pName);

    Py_DECREF(pModule);
    Py_DECREF(pName);

    Py_FinalizeEx();

    #ifdef VerboseCCamb
    printf("%s   Finalized Python.\n",indent);
    #endif
}

void get_pk(const char* params_file, double* grid, double z, int ng, double rl, int calls){
    getIndent(calls);
    #ifdef VerboseCCamb
    printf("%sFilling grid...\n",indent);
    #endif
    //double delK = (2*M_PI)/((double)ng);
    //double delK2 = delK*delK;
    double d = (2*M_PI)/rl;
    for (int i = 0; i < ng; i++){
        double l = i;
        if (i > ((ng/2)-1)){
            l = -(ng - i);
        }
        l *= d;
        for (int j = 0; j < ng; j++){
            double m = j;
            if (j > ((ng/2)-1)){
                m = -(ng - j);
            }
            m *= d;
            for (int k = 0; k < ng; k++){
                double n = k;
                if (k > ((ng/2)-1)){
                    n = -(ng - k);
                }
                n *= d;
                //grid[i*ng*ng + j*ng + k] = sqrt((i-ng/2)*(i-ng/2)*delK2 + (j-ng/2)*(j-ng/2)*delK2 + (k-ng/2)*(k-ng/2)*delK2);
                
                
                grid[i*ng*ng + j*ng + k] = sqrt(l*l + m*m + n*n);
            }
        }
    }
    #ifdef VerboseCCamb
    printf("%s   Filled grid.\n",indent);

    printf("%sGetting Functions...\n",indent);
    #endif

    // pDict is a borrowed reference 
    PyObject* pDict = PyModule_GetDict(pModule);

    // pFunc is also a borrowed reference get_pk
    PyObject* pFuncInit = PyDict_GetItemString(pDict, "initcambpy");
    
    PyObject* pGetPk = PyDict_GetItemString(pDict, "get_pk");

    PyObject* presult;

    PyObject* pPkResult;
    PyObject* args;

    #ifdef VerboseCCamb
    printf("%s   Got Functions.\n",indent);
    printf("%sCalling pFuncInit...\n",indent);
    #endif

    presult=PyObject_CallFunctionObjArgs(pFuncInit,NULL);
    PyErr_Print();

    #ifdef VerboseCCamb
    printf("%s   Called pFuncInit.\n",indent);

    printf("%sPacking into numpy array...\n",indent);
    #endif

    npy_intp dims[1];
    dims[0] = ng*ng*ng;
    int ND = 1;
    
    PyArrayObject* test = PyArray_SimpleNewFromData(ND, dims, NPY_DOUBLE, (void*)(grid));
    PyErr_Print();

    #ifdef VerboseCCamb
    printf("%s   Packed into numpy array.\n",indent);

    printf("%sCalling pGetPk...\n",indent);
    #endif
    args = PyTuple_Pack(5,PyFloat_FromDouble(z),test,PyFloat_FromDouble((double)ng),PyFloat_FromDouble(rl),_PyUnicode_FromASCII(params_file,strlen(params_file)*sizeof(char)));
    pPkResult = PyObject_CallObject(pGetPk,args);
    PyErr_Print();
    Py_DECREF(args);
    #ifdef VerboseCCamb
    printf("%s   Called pGetPk.\n",indent);
    #endif

    double* c_out = (double*)(PyArray_DATA(pPkResult));
    PyErr_Print();
    grid[0] = 0;
    for (int i = 0; i < ng*ng*ng; i++){
        grid[i] = c_out[i];
    }

}

void get_delta_and_dotDelta(const char* params_file, double z, double z1, double* delta, double* dotDelta, int calls){

    getIndent(calls);
    // pDict is a borrowed reference 
    PyObject* pDict = PyModule_GetDict(pModule);

    // pFunc is also a borrowed reference get_pk
    PyObject* pGetDeltaAndDotDelta = PyDict_GetItemString(pDict, "get_delta_and_dotDelta");
    
    //PyObject* pGetPk = PyDict_GetItemString(pDict, "get_pk");

    //PyObject* presult;

    PyObject* pPkResult;
    PyObject* args;

    #ifdef VerboseCCamb
    printf("%sCalling pGetDeltaAndDotDelta...\n",indent);
    #endif

    args = PyTuple_Pack(3,PyFloat_FromDouble(z),PyFloat_FromDouble(z1),_PyUnicode_FromASCII(params_file,strlen(params_file)*sizeof(char)));
    pPkResult = PyObject_CallObject(pGetDeltaAndDotDelta,args);
    PyErr_Print();
    Py_DECREF(args);
    #ifdef VerboseCCamb
    printf("%s   Called pGetDeltaAndDotDelta.\n",indent);
    #endif

    double* c_out = (double*)(PyArray_DATA(pPkResult));
    PyErr_Print();
    *delta = c_out[0];
    *dotDelta = c_out[1];
}