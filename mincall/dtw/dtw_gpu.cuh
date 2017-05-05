__global__
#include <cstdio>
#include <cmath>
#include <algorithm>    // std::min

template<typename T>
__device__ T sqr(T a) {
    return a*a;
}

template<typename T, int kernelSize>
__device__ T cost(T* A, T* B, int idx_a, int idx_b){
    return sqr(seq_a[idx_a] - seq_b[idx_b]);
}


template<typename T, int kernelSize>
__device__  __host__ void update_grad(T* da, T *db, int idx_a, int idx_b) {
    da[idx_a] += seq_a[idx_a] - seq_b[idx_b];
    db[idx_b] -= seq_a[idx_a] - seq_b[idx_b];
}


template<typename T, int kernelSize>
__global__ dtw_conv1d(T* signal, T* kernel, T *loss, T* gradSignal, T* gradKernel) {
    T* dp[kernelSize][kernelSize];
    T* signalBegin = signal; // some magic calc
    T* kernelBegin = kernel; // some magic calc
    T* da = gradSignal;
    T* db = gradKernel;

    dp[0][0] = cost(signalBegin, kernelBegin, 0, 0);

    #pragma unroll
    for (size_t i = 1; i < kernelSize; i++) {
        dp[i][0] = dp[i - 1][0] + cost(signalBegin, kernelBegin, i, 0);
    }

    #pragma unroll
    for (size_t i = 1; i < kernelSize; i++) {
        dp[0][i] = dp[0][i - 1] + cost(signalBegin, kernelBegin, 0, i);
    }

    #pragma unroll
    for (int i = 1; i < kernelSize; ++i)
        #pragma unroll
        for(int j = 1; j < kernelSize; ++j)
            dp[i][j] = cost(signalBegin, kernelBegin, i,j) + min(
                dp[i-1][j-1], min(
                    dp[i-1][j],
                    dp[i][j - 1]
                ));

    *loss = 0.5 * dp[kernelSize - 1][kernelSize - 1];
    // initialized outside, hopefully
    // for(int i = 0; i < n; ++i)
    //     da[i] = 0;
    //
    // for(int i = 0; i < m; ++i)
    //     db[i] = 0;

    int r = n - 1;
    int c = m - 1;
    int cr, cc;
    while (r > 0 && c > 0) {
        update_grad(da, db, r, c);
        // Techincally if there's two+ best dp grad w.r.t. cost is 0
        cr = r - 1;
        cc = c - 1;

        if (dp[r - 1][c] < dp[cr][cc]) {
            cr = r - 1;
            cc = c;
        }

        if (dp[r][c - 1] < dp[cr][cc]) {
            cr = r;
            cc = c - 1;
        }

        r = cr;
        c = cc;
    }

    for(;r > 0; --r) {
        update_grad(da, db, r, c);
    }

    for(;c > 0; --c) {
        update_grad(da, db, r, c);
    }
    update_grad(da, db, 0, 0);


}
