#include "IndexFlat.h"

void IndexFlatL2::add(int n, const float *x){
    xb.insert(xb.end(), x, x+(n*d));
    ntotal+=n;
}

void IndexFlatL2::search(int n, const float *x, int k, float *distances, int *labels){
    for(int i = 0; i<n; i++){//iterate over the entire query
        float min_distance = 1e9;
        int bestid = -1;
        for(int j= 0; j<ntotal; j++){//compare query against every vec in db
            float curr_distance = 0;
            for(int m=0; m<d; m++){//calc distance across all d dimensions 
                int ith_index = i*d+m;
                int jth_index = j*d+m;
                
                float diff=xb[jth_index]-x[ith_index];
                curr_distance+=(diff*diff);
            }
            if (curr_distance<min_distance){
                min_distance=curr_distance;
                bestid = j;
            }   
        }
        distances[i] = min_distance;
        labels[i] = bestid;
    }
}