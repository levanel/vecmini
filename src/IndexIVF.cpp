#include "IndexIVF.h"
#include "clustering.h"
#include <queue>

IndexIVF::IndexIVF(int d, int nbucket): d(d), nbucket(nbucket), router(d){
    memory.resize(nbucket);
};

void IndexIVF::train(int n, const float *x){
     if(trained) return;
     std::vector<float>centroids(nbucket*d);
     kmean_clustering(d, n, nbucket, x ,centroids.data());
     router.add(nbucket, centroids.data());
     trained=true;
}
void IndexIVF::add(int n, const float *x){
     if(!trained) return;
     std::vector<int> assign(n);
     std::vector<float> distances(n);
     router.search(n,x,1,distances.data(), assign.data());
     for(int i =0; i<n; i++){
        int bucketid= assign[i];
        memory[bucketid].insert(memory[bucketid].end(),x+(i*d), x+(i*d)+d);
    }
    ntotal+=n;
}
void IndexIVF::search(int n, const float* x, int k,float *distances, int *labels){
    std::vector<int> assign(n);
    std::vector<float> centroids_distance(n);
    router.search(n,x,1,centroids_distance.data(), assign.data());
    for(int i = 0; i<n; i++){
        int bucketid = assign[i];
        int vectorinmemo = memory[bucketid].size()/d;
        if(vectorinmemo==0)continue;
        const float *specquer = x+(i*d);
        const float *bucketdata=  memory[bucketid].data();
        
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> pq;
        for(int j = 0; j<vectorinmemo; j++){
            float currcosine = 0;
            for(int m = 0; m<d; m++){
                currcosine+=(bucketdata[j*d+m]*specquer[m]);
            }

            if(pq.size()<k){
                pq.push({currcosine, j});
            }else{
                if(currcosine>pq.top().first){
                    pq.pop();
                    pq.push({currcosine, j});
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
