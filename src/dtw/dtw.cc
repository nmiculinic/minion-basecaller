#include <cstdio>
#include <cmath>
#include <algorithm>    // std::min

const int n = 3;
const int m = 4;
using std::min;
using std::max;

void print_arr(double arr[n][m]) {
    for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; j++) {
            printf("%10.6lf", arr[i][j]);
        }
        printf("\n");
    }
}

double eps = 1e-7;
double seq_a[n] = {1, 2, 3};
double seq_b[m] = {1.1, 1.7, 2.3, 2.5};

double sqr(double a) {
    return a*a;
}

double cost(int idx_a, int idx_b){
    return 0.5 * sqr(seq_a[idx_a] - seq_b[idx_b]);
}

double da[n] = {0};
double db[m] = {0};

void print_da() {
    printf("da: ");
    for(int i = 0; i < n; ++i)
        printf("%7.5lf ", da[i]);

    printf("\n");
    printf("db: ");
    for(int i = 0; i < m; ++i)
        printf("%7.5lf ", db[i]);

    printf("\n");
}

void update_grad(double* da, double *db, int idx_a, int idx_b) {
    // printf("%d %d\n", idx_a, idx_b);
    da[idx_a] += seq_a[idx_a] - seq_b[idx_b];
    db[idx_b] -= seq_a[idx_a] - seq_b[idx_b];
}

double calc_loss(bool print) {
    double dp[n][m] = {0};
    dp[0][0] = cost(0, 0);
    for (size_t i = 1; i < n; i++) {
        dp[i][0] = dp[i - 1][0] + cost(i, 0);
    }

    for (size_t i = 1; i < m; i++) {
        dp[0][i] = dp[0][i - 1] + cost(0, i);
    }

    for (int i = 1; i < n; ++i)
        for(int j = 1; j < m; ++j)
            dp[i][j] = cost(i,j) + min(
                dp[i-1][j-1], min(
                    dp[i-1][j],
                    dp[i][j - 1]
                ));

    for(int i = 0; i < n; ++i)
        da[i] = 0;

    for(int i = 0; i < m; ++i)
        db[i] = 0;

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
    if (print) {
        print_arr(dp);
        printf("\n");
        print_da();
    }
    return dp[n - 1][m - 1];
}

int main(int argc, char const *argv[]) {
    calc_loss(true);
    printf("\n");
    for(int i = 0; i < n; ++i) {
        seq_a[i] += eps;
        double ap = calc_loss(false);
        seq_a[i] -= 2 * eps;
        double ad = calc_loss(false);
        seq_a[i] += eps;
        double num_der = (ap-ad)/(2*eps);
        calc_loss(false);
        printf("%d %7.5lf %6.3f\n", i, num_der ,fabs(num_der - da[i]));
    }
    printf("\n");

    for(int i = 0; i < m; ++i) {
        seq_b[i] += eps;
        double ap = calc_loss(false);
        seq_b[i] -= 2 * eps;
        double ad = calc_loss(false);
        seq_b[i] += eps;
        calc_loss(false);
        printf("%d %.5lf %.3f\n", i, (ap-ad)/(2*eps), fabs((ap-ad)/(2*eps) - db[i]));
    }
    return 0;
}
