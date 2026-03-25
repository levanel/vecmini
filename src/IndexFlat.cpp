#include "IndexFlat.h"
#include <queue>
#include <vector>
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
            for(int m=0; m<d; m++){//calc distance across all d dimensions 
                int ith_index = i*d+m;
                int jth_index = j*d+m;
                
                float diff=xb[jth_index]-x[ith_index];
                curr_distance+=(diff*diff);

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