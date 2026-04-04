#include <IndexPQ.h>
#include <vector>
#include "clustering.h"
#include<immintrin.h>

IndexPQ::IndexPQ(int d, int m):d(d), m(m){
    k_sub = 256;
    d_sub = d/m;
    centroids.resize(m*d_sub*k_sub);
}; 

void IndexPQ::train(int n, const float *x){
    if(trained) return;
    std::vector<float>train_data(n*d_sub);
    for(int i=0 ; i<m; i++){
        for(int row = 0; row<n; row++){
            const float* source_id = x+(row*d)+(i*d_sub);
            float* dest_id = train_data.data()+(row*d_sub);
            for(int j = 0; j<d_sub; j++){
                dest_id[j]= source_id[j];
            }
        }
        kmean_clustering(d_sub, n, k_sub, train_data.data() , centroids.data()+(i*d_sub*k_sub));
    }    
    trained=true;
}

void IndexPQ::encode(const float *x, uint8_t* out){
    if(!trained)return;
    for(int i =0; i<m; i++){
        const float *query_chunk = x + (i*d_sub);
        float mindistance = 1e9;
        int bestid = 0;
        for(int id=0; id<k_sub; id++){
            const float* centroid_chunk = centroids.data()+(i*k_sub*d_sub)+(id*d_sub);
            float dist = 0;
            for(int j =0; j<d_sub; j++){
                float diff = query_chunk[j]- centroid_chunk[j];
                  dist += diff*diff;
            }
            if(dist<mindistance){
                mindistance = dist;
                bestid = id;
            }
        }
        out[i] = bestid;
    }
}
//precalc all distance for query and centroid 
void IndexPQ::compute_distance_table(const float *query, float *outable){
    for(int i =0; i<m; i++){
        const float* query_slice = query+(i*d_sub);
        for(int j = 0; j<k_sub; j++){
            float dist = 0; 
            const float *offset= centroids.data()+(i*k_sub*d_sub) + (j*d_sub); 
            /*for(int k = 0;k<d_sub; k++){
                float diff = offset[k]-query_slice[k];
                dist+=diff*diff;
            }*/
             __m256 sumvec = _mm256_setzero_ps();
            for(int k =0; k<d_sub; k+=8){
               __m256 offvec= _mm256_loadu_ps(&offset[k]); 
               __m256 querslice= _mm256_loadu_ps(&query_slice[k]);
               __m256 diffvec =  _mm256_sub_ps(offvec,querslice);
               sumvec = _mm256_fmadd_ps(diffvec, diffvec, sumvec);
            }
            float unpacked[8];
            _mm256_storeu_ps(unpacked, sumvec);
                   dist=unpacked[0]+unpacked[1]+
                        unpacked[2]+unpacked[3]+
                        unpacked[4]+unpacked[5]+
                        unpacked[6]+unpacked[7];
            outable[(i*k_sub)+j] = dist;
        }
    }
}

