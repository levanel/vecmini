#pragma once
#include <vector>

class IndexFlatL2{
    int d;//dimension of vector
    int ntotal=0;//no of vector in the db                   
    std::vector<float>xb;

    public:
        IndexFlatL2(int d) : d(d) {}
        // ingests 'n'vectors from a raw memory pointer 'x' into the database
        void add(int n, const float *x);
        //k->how many nearest neghbour we want
        // ans is saved in distances and labels
        void search(int n, const float *x, int k, float *distances, int* labels);
};