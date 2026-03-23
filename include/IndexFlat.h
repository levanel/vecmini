#pragma once
#include <vector>
class IndexFlatL2{
    int d;//dimension of vector
    int ntotal=0;//no of vector in the db                   
    std::vector<float>xb;

    public:
        IndexFlatL2(int d) : d(d) {}
        void add(int n, const float *x);
        void search(int n, const float *x, int k, float *distances, int* labels);
};