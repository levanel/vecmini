#include "IndexIVFPQ.h"
#include "IndexIVF.h"
#include "clustering.h"
#include <queue>
#include <iostream>
#include <immintrin.h>
#include <random>
#include <cstring>
IndexIVFPQ::IndexIVFPQ(int d, int nbucket, int m): d(d), m(m), nbucket(nbucket), router(d), pq(d, m){
    codes.resize(nbucket);
    ids.resize(nbucket);
};

void IndexIVFPQ::train(int n, const float *x, bool subsampling, int seed){
    if(trained)return;
    coarse_centroids.resize(nbucket*d);
    
    int maxtrain = 150000;
    if(n>maxtrain && subsampling){
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int>dis(0,n-1);
        std::vector<float> sample_buffer(maxtrain * d);
        for(int i=0; i<maxtrain; i++){
            int randval = dis(gen);
            std::memcpy(&sample_buffer[i*d],
                         &x[randval * d],
                         d * sizeof(float));
        }
        kmean_clustering(d, maxtrain, nbucket, sample_buffer.data(), coarse_centroids.data(), seed);
    }else{kmean_clustering(d, n, nbucket, x, coarse_centroids.data(), seed);}    

    router.add(nbucket, coarse_centroids.data());
    std::vector<float>residuals(n*d);
    std::vector<float> distances(n);
    std::vector<int> labels(n);
    router.search(n,x,1,distances.data(), labels.data());
    for(int i = 0;i<n; i++){
        int drawerid = labels[i];
        for(int j = 0; j<d; j++){
            residuals[(i*d)+j] = x[(i*d)+j] - coarse_centroids[(drawerid*d)+j];  
        }
    }
    pq.train(n, residuals.data(), subsampling, seed);
    trained = true;
}
void IndexIVFPQ::add(int n, const float *x, const uint64_t* xids){
    if (!trained) return;
    std::vector<float>residuals(n*d);
    std::vector<float> distances(n);
    std::vector<int> labels(n);
    router.search(n,x,1,distances.data(), labels.data());
    std::cout << "expected centroids size: " << nbucket * d << std::endl;
std::cout << "actual centroids size: " << coarse_centroids.size() << std::endl;
std::cout << "codes vector size: " << codes.size() << std::endl;   
    for(int i = 0;i<n; i++){
        int drawerid = labels[i];
        for(int j = 0; j<d; j++){
            residuals[(i*d)+j] = x[(i*d)+j]-coarse_centroids[(drawerid*d)+j];
        }
        std::vector<uint8_t> zipvect(m);
        pq.encode(residuals.data()+(i*d), zipvect.data());
        codes[drawerid].insert(codes[drawerid].end(), zipvect.begin(), zipvect.end());
        ids[drawerid].push_back(xids[i]);
    }
}
void IndexIVFPQ::search(int n, const float *query, int k, int nprobe, float* distances, int64_t* labels){
    std::vector<int> assign(n*nprobe);
    std::vector<float> coarse_distances(n*nprobe);
    router.search(n,query, nprobe, coarse_distances.data(),assign.data());
    for(int i = 0; i<n; i++){
        std::priority_queue<std::pair<float, int>> max_heap;
        std::vector<float> query_residual(d);
        for(int p=0; p<nprobe; p++){
            int drawerid = assign[(i*nprobe)+p];
            /*for(int j = 0; j<d; j++){
                query_residual[j] = query[(i*d)+j] - coarse_centroids[(drawerid*d)+j];
            }
            */

            for(int j=0; j<d; j+=8){
                __m256 ccvec= _mm256_loadu_ps(&coarse_centroids[(drawerid*d)+j]);
                __m256 qrvec= _mm256_loadu_ps(&query[(i*d)+j]);
                __m256 diffvec = _mm256_sub_ps(qrvec,ccvec);
                _mm256_storeu_ps(&query_residual[j], diffvec);
            }   
            

            std::vector<float> distance_table(m*256);
            pq.compute_distance_table(query_residual.data(), distance_table.data());
            for(int v = 0; v<codes[drawerid].size()/m; v++){
                float totaldistance =0.0;
                for(int m_idx = 0; m_idx<m; m_idx++){
                    int centroid_id = codes[drawerid][(v*m)+m_idx];
                    totaldistance+=distance_table[centroid_id+(m_idx*256)];
                }
                if(max_heap.size()<k){
                    max_heap.push({totaldistance, ids[drawerid][v]});
                }else{
                    if(totaldistance<max_heap.top().first){
                        max_heap.pop();
                        max_heap.push({totaldistance, ids[drawerid][v]});
                    }
                }
            }
        }
        float *subdist = distances+(i*k);
        int64_t *sublbs = labels+(i*k);
        int count = max_heap.size();
        for(int c = count-1; c>=0; c--){
            subdist[c] = max_heap.top().first;
            sublbs[c] = max_heap.top().second;
            max_heap.pop();            
        }
        for(int fod = count; fod<k; fod++){
            subdist[fod]=-1.0;
            sublbs[fod]=-1;
        }
    }
}
