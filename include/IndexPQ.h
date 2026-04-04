#pragma once
#include <vector>
#include <cstdint>
class IndexPQ{
private: 
    int d; 
    int m; 
    int k_sub;
    int d_sub;

    std::vector<float> centroids;
    bool trained  = false;
public: 
    IndexPQ(int d, int m);
    void train(int n, const float* x);
    void encode(const float *x, uint8_t* out);
    void compute_distance_table(const float *query, float *outable);
};