#pragma once
#include <vector>
#include "IndexFlat.h"

class IndexIVF {
private: // The vault (Only the database can touch these)
    int d; 
    int nbucket; 
    int ntotal = 0;
    bool trained = false;
    
    IndexFlatL2 router; 
    std::vector<std::vector<float>> memory;

public: // The interface (Your benchmark script is allowed to use these)
    IndexIVF(int d, int nbucket);    
    void train(int n, const float *x);
    void add(int n, const float *x);
    void search(int n, const float* x, int k, float *distances, int *labels);
};