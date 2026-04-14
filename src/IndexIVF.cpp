#include "IndexIVF.h"
#include "clustering.h"
#include <queue>
#include <iostream>
#include<immintrin.h>
IndexIVF::IndexIVF(int d, int nbucket): d(d), nbucket(nbucket), router(d){
    memory.resize(nbucket);
    memory_ids.resize(nbucket);
};

void IndexIVF::train(int n, const float *x){
     if(trained) return;
     std::vector<float>centroids(nbucket*d);

     //remove seed
     kmean_clustering(d, n, nbucket, x ,centroids.data(),1);
     router.add(nbucket, centroids.data());
     trained=true;
}

void IndexIVF::add(int n, const float *x, const uint64_t*xids){
     if(!trained) return;
     std::vector<int> assign(n);
     std::vector<float> distances(n);
     router.search(n,x,1,distances.data(), assign.data());
     for(int i =0; i<n; i++){
        int bucketid= assign[i];
        memory[bucketid].insert(memory[bucketid].end(),x+(i*d), x+(i*d)+d);
        //for metadata
        memory_ids[bucketid].push_back(xids[i]);
    }
    ntotal+=n;
}
void IndexIVF::search(int n, const float* x, int k, int nprobe, const uint8_t *bitmask, float *distances, int *labels){
    std::vector<int>assign(n*nprobe);
    std::vector<float> centroids_distance(n*nprobe);

    router.search(n,x,nprobe,centroids_distance.data(), assign.data());
    for(int i = 0; i<n; i++){
        //std::unordered_set <uint64_t> set;
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> pq;
        const float *specquer = x+(i*d);
        for(int p= 0; p<nprobe; p++){
            int bucketid = assign[i*nprobe+p];
            int vectorinmemo = memory[bucketid].size()/d;
            if(vectorinmemo==0)continue;
            const float *bucketdata=  memory[bucketid].data();

        for(int j = 0; j<vectorinmemo; j++){

            uint64_t global_id = memory_ids[bucketid][j];
            //redundant now
            /*
            //skip the calc if bimask doesnt exist
            if(set.count(global_id)){
                continue;
            }
            set.insert(global_id);
*/  
            if(bitmask!=nullptr && bitmask[global_id]==0) {
                continue;
            }
            //removed this for simd

            //benchmark for standard cosine calc->
            //nullptr: 6.32857
            //bitmask: 4.60353

            //after simd
            //nullptr: 1.3298
            //bitmask: 0.918149

            //added simd
            float currcosine = 0;
            int m = 0;

            if(j+1<vectorinmemo){
                _mm_prefetch((const char*)&bucketdata[(j+1)*d], _MM_HINT_T0);
            }
            __m256 sumvec = _mm256_setzero_ps();

            /*for(int m = 0; m<d; m++){
                currcosine+=(bucketdata[j*d+m]*specquer[m]);
            }*/
            for(; m+7<d; m+=8){
                //load 8float from the db chunk
                __m256 dbvec=  _mm256_loadu_ps(&bucketdata[j*d+m]);
                __m256 qvec=  _mm256_loadu_ps(&specquer[m]);
                sumvec = _mm256_fmadd_ps(dbvec, qvec, sumvec);
            }
            
            __m128 upper = _mm256_extractf128_ps(sumvec, 1);
            __m128 lower = _mm256_extractf128_ps(sumvec, 0);

            __m128 sumbound = _mm_add_ps(upper, lower);
            __m128 shifted = _mm_movehl_ps(sumbound, sumbound);
            __m128 current = _mm_add_ps(sumbound, shifted);
            __m128 shuffled = _mm_shuffle_ps(current, current, 1);
            __m128 finalsum = _mm_add_ps(current, shuffled);
            currcosine = _mm_cvtss_f32(finalsum);

/*
            float sumarr[8];
            _mm256_storeu_ps(sumarr,sumvec);
            currcosine= sumarr[0]+sumarr[1]+
                        sumarr[2]+sumarr[3]+
                        sumarr[4]+sumarr[5]+
                        sumarr[6]+sumarr[7];
            //cleanup       
  */
            if(pq.size()<k){
                pq.push({currcosine, global_id});
            }else{
                if(currcosine>pq.top().first){
                    pq.pop();
                    pq.push({currcosine, global_id});
                }
            }
        }
    }
        float *speldist = distances+(i*k);
        int *spelbs = labels+(i*k);
        int count = pq.size();
        for(int c = count-1; c>=0; c--){
            speldist[c]= pq.top().first;
            spelbs[c]= pq.top().second;
            pq.pop(); 
        }
        for(int step = count; step<k; step++){
            speldist[step]=-1.0; 
            spelbs[step]= -1;
        }
    }
}
