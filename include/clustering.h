#pragma once
#include<vector>

void kmean_clustering(
    int d,
    int n,
    int k,
    const float *x,
    float *centroids,
    int seed
);  