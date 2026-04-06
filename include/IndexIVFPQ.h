#pragma once
#include<vector>
#include<cstdint>
#include "IndexPQ.h"
#include "IndexFlat.h"
#include <cstddef>
class IndexIVFPQ{
private:
    int d;
    int m;//bitquant 
    int nbucket; //no of centroid
    int ntotal; //no of vector index
    bool trained;
    size_t nprobe;//how many voronoi i should look at
    IndexFlatL2 router;
    IndexPQ pq;
    std::vector<float>coarse_centroids;
    std::vector<std::vector<uint8_t>>codes;
    std::vector<std::vector<int64_t>>ids;

public: 
    IndexIVFPQ(int d, int nbucket, int m);
    void train(int n, const float *x, bool subsampling);
    void add(int n, const float *x, const uint64_t* xids);
    void search(int n, const float *query, int k, int nprobe, float* distances, int64_t* labels);
};


