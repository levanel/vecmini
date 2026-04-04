#include "IndexFlat.h"
#include <queue>
#include <vector>
#include<immintrin.h>

void IndexFlatL2::add(int n, const float *x){
    xb.insert(xb.end(), x, x+(n*d));
    ntotal+=n;
}

void IndexFlatL2::search(int n, const float *x, int k, float *distances, int *labels){
    for(int i = 0; i<n; i++){//iterate over the entire query
        //old stuff
            //float min_distance = 1e9;
            //int bestid = -1;
        std::priority_queue<std::pair<float, int>> pq;

        for(int j= 0; j<ntotal; j++){//compare query against every vec in db
            float curr_distance = 0;
            int m = 0;
            
            __m256 sumvec = _mm256_setzero_ps();
            
            const float* current_db_vec = &xb[j * d];
            const float* current_q_vec = &x[i * d];

            for(; m + 7 < d; m += 8){
                __m256 dbvec = _mm256_load_ps(&current_db_vec[m]);
                __m256 qvec = _mm256_load_ps(&current_q_vec[m]);
                
                __m256 diff = _mm256_sub_ps(dbvec, qvec);
                
                sumvec = _mm256_fmadd_ps(diff, diff, sumvec);
            }
            
            float sumarr[8];
            _mm256_storeu_ps(sumarr, sumvec);
            curr_distance = sumarr[0] + sumarr[1] + sumarr[2] + sumarr[3] + 
                            sumarr[4] + sumarr[5] + sumarr[6] + sumarr[7];
            
            for(; m < d; m++){
                float dist = current_db_vec[m] - current_q_vec[m];
                curr_distance += (dist * dist);
            }
            /*
            if (curr_distance<min_distance){
                min_distance=curr_distance;
                bestid = j;
            } */
             
            

            if(pq.size()<k){
                pq.push({curr_distance,j});
            }else{
                if(curr_distance<pq.top().first){
                    pq.pop();
                    pq.push({curr_distance,j});
                }
            }
        }
        /*
        distances[i] = min_distance;
        labels[i] = bestid; */


        //standard for loop cannot handle garbage values. 
        //for that we need 2 seperate for loop, one that handles the queue content properly
        int count=pq.size();
        for(int c = count-1; c>=0; c--){
            distances[i*k+c] = pq.top().first;
            labels[i*k+c] = pq.top().second;
            pq.pop();
        }
        for(int step=count; step<k; step++){
            distances[i*k+step] = -1.0;
            labels[i*k+step] = -1;
        }
    }
}