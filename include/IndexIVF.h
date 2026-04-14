#pragma once
#include <vector>
#include "IndexFlat.h"
#include <cstdint>

class IndexIVF {
private: 
    int d; 
    int nbucket; 
    int ntotal = 0;
    bool trained = false;
    
    IndexFlatL2 router; 
    std::vector<std::vector<float>> memory;
    std::vector<std::vector<uint64_t>> memory_ids;

public: // The interface (Your benchmark script is allowed to use these)
    IndexIVF(int d, int nbucket);    
    void train(int n, const float *x);
    void add(int n, const float *x, const uint64_t*xids);
    void search(int n, const float* x, int k, int nprobe, const uint8_t *bitmask, float *distances, int *labels);
};