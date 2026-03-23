#include<vector>
#include "clustering.h" 
#include "IndexFlat.h"
void kmean_clustering(int d, int n, int k, const float *x, float *centroids){
    for(int i = 0; i<k*d; i++){
        centroids[i]=x[i];
    }
    int niter = 10;
    std::vector<int> assign(n);
    std::vector<float> distances(n);
    for(int iter = 0; iter<niter; iter++){
        IndexFlatL2 index(d);
        index.add(k,centroids);
        index.search(n,x,1,distances.data(), assign.data());
        std::vector<float> newcentroid(k*d,0.0);
        std::vector<int> counts(k,0);
        for(int i = 0; i<n; i++){
            int c = assign[i];
            counts[c]+=1;
            for(int m =0; m<d; m++){
                newcentroid[c*d+m] += x[i*d+m];
            }
        }
        for(int c = 0; c<k; c++){
            if (counts[c]>0){
                for(int m = 0; m<d; m++){
                    centroids[c*d+m] = newcentroid[c*d+m]/counts[c];
                }
            }
        }
    }
}