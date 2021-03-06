#include "util.h"

namespace mips {

timeval g_start_time;				// global param: start time
timeval g_end_time;					// global param: end time

float   g_memory    = -1.0f;		// global param: estimated memory usage (MB)
float   g_indextime = -1.0f;		// global param: indexing time (seconds)

float   g_runtime   = -1.0f;		// global param: running time (ms)
float   g_ratio     = -1.0f;		// global param: overall ratio
float   g_recall    = -1.0f;		// global param: recall (%)

// -----------------------------------------------------------------------------
void create_dir(					// create dir if the path exists
	char *path)							// input path
{
	int len = (int) strlen(path);
	for (int i = 0; i < len; ++i) {
		if (path[i] == '/') {
			char ch = path[i + 1];
			path[i + 1] = '\0';
									// check whether the directory exists
			int ret = access(path, F_OK);
			if (ret != 0) {			// create the directory
				ret = mkdir(path, 0755);
				if (ret != 0) {
					printf("Could not create directory %s\n", path);
				}
			}
			path[i + 1] = ch;
		}
	}
}

// -----------------------------------------------------------------------------
int read_txt_data(					// read data (text) from disk
	int   n,							// number of data objects
	int   d,							// dimensionality
	bool  flag,							// true - data; false - query
	const char *fname,					// address of data set
	float **data,						// data objects (return)
	float **norm_d)						// l2-norm of data objects (return)
{
	gettimeofday(&g_start_time, NULL);
	FILE *fp = fopen(fname, "r");
	if (!fp) {
		printf("Could not open %s\n", fname);
		return 1;
	}

	int i = 0;
	int j = 0;
	while (!feof(fp) && i < n) {
		float tmp = 0.0f;
		memset(norm_d[i], 0.0f, NORM_K * SIZEFLOAT);

		fscanf(fp, "%d", &j);
		for (j = 0; j < d; ++j) {
			fscanf(fp, " %f", &tmp);
			data[i][j] = tmp;

			norm_d[i][0] += tmp*tmp;
			for (int t = 1; t < NORM_K; ++t) {
				if (j < 8*t) norm_d[i][t] += tmp*tmp;
			}
		}
		fscanf(fp, "\n");

		for (int t = 1; t < NORM_K; ++t) {
			norm_d[i][t] = sqrt(norm_d[i][0] - norm_d[i][t]);
		}
		norm_d[i][0] = sqrt(norm_d[i][0]);
		++i;
	}
	assert(feof(fp) && i == n);
	fclose(fp);

	gettimeofday(&g_end_time, NULL);
	float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
		(g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
	
	if (flag) printf("Read Data : %f Seconds\n", running_time);
	else printf("Read Query: %f Seconds\n", running_time);

	return 0;
}

int calculate_k_norm(
        int d,//dimensionality
        float *data,//input data
        float *norm_d// l2-norm (return)
        ){
    memset(norm_d, 0.0f, NORM_K * SIZEFLOAT);
    float tmp = 0.0f;
    for (int j = 0; j < d; ++j) {
        tmp = data[j];

        norm_d[0] += SQR(tmp);
        for (int t = 1; t < NORM_K; ++t) {
            if (j < 8 * t) norm_d[t] += SQR(tmp);
        }
    }
    for (int t = 1; t < NORM_K; ++t) {
        norm_d[t] = sqrt(norm_d[0] - norm_d[t]);
    }
    norm_d[0] = sqrt(norm_d[0]);
    return 0;
}
// -----------------------------------------------------------------------------
int read_bin_data(					// read data (binary) from disk
	int   n,							// number of data objects
	int   d,							// dimensionality
	bool  flag,							// true - data; false - query
	const char *fname,					// address of data
	float **data,						// data objects (return)
	float **norm_d)						// l2-norm of data objects (return)
{
	gettimeofday(&g_start_time, NULL);
	FILE *fp = fopen(fname, "rb");
	if (!fp) {
		printf("Could not open %s\n", fname);
		return 1;
	}

	int i = 0;
	while (!feof(fp) && i < n) {
		fread(data[i], SIZEFLOAT, d, fp);

		// calc norm_d
        calculate_k_norm(d,data[i],norm_d[i]);
		++i;
	}
	fclose(fp);

	gettimeofday(&g_end_time, NULL);
	float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
		(g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

	if (flag) printf("Read Data : %f Seconds\n", running_time);
	else printf("Read Query: %f Seconds\n", running_time);

	return 0;
}

// -----------------------------------------------------------------------------
int read_ground_truth(				// read ground truth results from disk
	int qn,								// number of query objects
	const char *fname,					// address of truth set
	Result **R)							// ground truth results (return)
{
	gettimeofday(&g_start_time, NULL);
	FILE *fp = fopen(fname, "r");
	if (!fp) {
		printf("Could not open %s\n", fname);
		return 1;
	}

	int tmp1 = -1;
	int tmp2 = -1;
	fscanf(fp, "%d %d\n", &tmp1, &tmp2);
	assert(tmp1 == qn && tmp2 == MAXK);

	for (int i = 0; i < qn; ++i) {
		for (int j = 0; j < MAXK; ++j) {
			fscanf(fp, "%d %f ", &R[i][j].id_, &R[i][j].key_);
		}
		fscanf(fp, "\n");
	}
	fclose(fp);

	gettimeofday(&g_end_time, NULL);
	float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
		(g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
	printf("Read Truth: %f Seconds\n\n", running_time);

	return 0;
}

// -----------------------------------------------------------------------------
void write_pre_recall(				// write precision-recall curves to disk
	FILE  *fp,							// file pointer
	const float **pre,					// precision 
	const float **recall)				// recall
{
	for (int r = 0; r < MAX_ROUND; ++r) {
		printf("Top-%d\t\tRecall\t\tPrecision\n", TOPK[r]);
		fprintf(fp, "Top-%d\tRecall\t\tPrecision\n", TOPK[r]);
		
		for (int t = 0; t < MAX_T; ++t) {
			printf("%4d\t\t%.2f\t\t%.2f\n", tMIPs[t], recall[r][t], pre[r][t]);
			fprintf(fp, "%d\t%f\t%f\n", tMIPs[t], recall[r][t], pre[r][t]);
		}
		printf("\n");
		fprintf(fp, "\n");
	}
	printf("\n");
	fprintf(fp, "\n");
}

// -----------------------------------------------------------------------------
float calc_inner_product(			// calc inner product
	int   dim,							// dimension
	const float *p1,					// 1st point
	const float *p2)					// 2nd point
{
	float ret = 0.0f;
	for (int i = 0; i < dim; ++i) {
		ret += p1[i] * p2[i];
	}
	return ret;
}

// -----------------------------------------------------------------------------
float calc_inner_product(			// calc inner product
	int   dim,							// dimension
	float threshold,					// threshold
	const float *p1,					// 1st point
	const float *norm1,					// l2-norm of 1st point
	const float *p2,					// 2nd point
	const float *norm2) 				// l2-norm of 2nd point
{
	float ip = 0.0f;
	int base = 0;
	for (int t = 1; t < NORM_K; ++t) {
		int end = base + 8;
		for (int i = base; i < end; ++i) {
			ip += p1[i] * p2[i];
		}
		if (ip + norm1[t]*norm2[t] <= threshold) return ip; 
		base += 8;
	}
	for (int i = base; i < dim; ++i) {
		ip += p1[i] * p2[i];
	}
	return ip;

	// unsigned d = dim & ~unsigned(7);
	// const float *aa = p1, *end_a = aa + d;
	// const float *bb = p2, *end_b = bb + d;

	// float r = 0.0f;
	// float r0, r1, r2, r3, r4, r5, r6, r7;

	// const float *a = aa, *b = bb;
	// int t = 1;
	// for (; a < end_a; a += 8, b += 8) {
	// 	__builtin_prefetch(a+32, 0, 3);
	// 	__builtin_prefetch(b+32, 0, 0);

	// 	r0 = a[0] * b[0];
	// 	r1 = a[1] * b[1];
	// 	r2 = a[2] * b[2];
	// 	r3 = a[3] * b[3];
	// 	r4 = a[4] * b[4];
	// 	r5 = a[5] * b[5];
	// 	r6 = a[6] * b[6];
	// 	r7 = a[7] * b[7];

	// 	r += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;

	// 	if (t < NORM_K && r+norm1[t]*norm2[t] <= threshold) return r;
	// 	++t;
	// }
	// __builtin_prefetch(a, 0, 3);
	// __builtin_prefetch(b, 0, 0);

	// r0 = r1 = r2 = r3 = r4 = r5 = r6 = r7 = 0.0f;
	// switch (dim & 7) {
	// 	case 7: r6 = a[6] * b[6];
	// 	case 6: r5 = a[5] * b[5];
	// 	case 5: r4 = a[4] * b[4];
	// 	case 4: r3 = a[3] * b[3];
	// 	case 3: r2 = a[2] * b[2];
	// 	case 2: r1 = a[1] * b[1];
	// 	case 1: r0 = a[0] * b[0];
	// }
	// r += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
	
	// return r;
}

// -----------------------------------------------------------------------------
float calc_l2_sqr(					// calc L2 square distance
	int   dim,							// dimension
	float threshold,					// threshold
	const float *p1,					// 1st point
	const float *p2)					// 2nd point
{
	unsigned d = dim & ~unsigned(7);
	const float *aa = p1, *end_a = aa + d;
	const float *bb = p2, *end_b = bb + d;

	__builtin_prefetch(aa, 0, 3);
	__builtin_prefetch(bb, 0, 0);

	float r = 0.0f;
	float r0, r1, r2, r3, r4, r5, r6, r7;

	const float *a = end_a, *b = end_b;

	r0 = r1 = r2 = r3 = r4 = r5 = r6 = r7 = 0.0f;
	switch (dim & 7) {
		case 7: r6 = SQR(a[6] - b[6]);
		case 6: r5 = SQR(a[5] - b[5]);
		case 5: r4 = SQR(a[4] - b[4]);
		case 4: r3 = SQR(a[3] - b[3]);
		case 3: r2 = SQR(a[2] - b[2]);
		case 2: r1 = SQR(a[1] - b[1]);
		case 1: r0 = SQR(a[0] - b[0]);
	}

	a = aa; b = bb;
	for (; a < end_a; a += 8, b += 8) {
		__builtin_prefetch(a+32, 0, 3);
		__builtin_prefetch(b+32, 0, 0);

		r += (r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7);
		if (r > threshold) return r;

		r0 = SQR(a[0] - b[0]);
		r1 = SQR(a[1] - b[1]);
		r2 = SQR(a[2] - b[2]);
		r3 = SQR(a[3] - b[3]);
		r4 = SQR(a[4] - b[4]);
		r5 = SQR(a[5] - b[5]);
		r6 = SQR(a[6] - b[6]);
		r7 = SQR(a[7] - b[7]);
	}
	r += (r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7);
	
	return r;
}

// -----------------------------------------------------------------------------
float calc_ratio(					// calc overall ratio
	int   k,							// top-k value
	const Result *R,					// ground truth results 
	MaxK_List *list)					// results returned by algorithms
{
	// add penalty if list->size() < k
	if (list->size() < k) return MAXREAL;

	// consider geometric mean instead of arithmetic mean for overall ratio
	float sum = 0.0f, r = -1.0f;
	for (int j = 0; j < k; ++j) {
		if (fabs(list->ith_key(j) - R[j].key_) < FLOATZERO) r = 1.0f;
		else r = (list->ith_key(j) + 1e-9) / (R[j].key_ + 1e-9);
		sum += log(r);
	}
	return pow(E, sum / k);
}

// -----------------------------------------------------------------------------
float calc_recall(					// calc recall of mip results
	int   k,							// top-k value
	const Result *R,					// ground truth results 
	MaxK_List *list)					// results returned by algorithms
{
	int i = list->size() - 1;
	int last = k - 1;
	while (i >= 0 && R[last].key_ - list->ith_key(i) > FLOATZERO) {
		--i;
	}
	return (i + 1) * 100.0f / k;
}

// -----------------------------------------------------------------------------
int get_hits(						// get the number of hits between two ID list
	int   k,							// top-k value
	int   t,							// top-t value
	const Result *R,					// ground truth results 
	MaxK_List *list)					// results returned by algorithms
{
	int i = k - 1;
	int last = t - 1;
	while (i >= 0 && R[last].key_ - list->ith_key(i) > FLOATZERO) --i;

	return MIN(t, i + 1);
}

// -----------------------------------------------------------------------------
int norm_distribution(				// analyse norm distribution of data
	int   n,							// number of data objects
	int   d,							// dimensionality
	const float **data,					// data objects
	const float **norm_d,				// l2-norm of data objects
	const char  *out_path)				// output path
{
	// -------------------------------------------------------------------------
	//  find max l2-norm of all data objects
	// -------------------------------------------------------------------------
	gettimeofday(&g_start_time, NULL);
	float max_norm = MINREAL;
	for (int i = 0; i < n; ++i) {
		if (norm_d[i][0] > max_norm) max_norm = norm_d[i][0];
	}

	// -------------------------------------------------------------------------
	//  get the percentage of frequency of norm
	// -------------------------------------------------------------------------
	int m = 25;
	float interval = max_norm / m;
	printf("m = %d, max_norm = %f, interval = %f\n", m, max_norm, interval);

	std::vector<int> freq(m, 0);
	for (int i = 0; i < n; ++i) {
		int id = (int) ceil(norm_d[i][0] / interval) - 1;
		if (id < 0)  id = 0;
		if (id >= m) id = m - 1;
		freq[id]++;
	}

	// -------------------------------------------------------------------------
	//  write norm distribution
	// -------------------------------------------------------------------------
	char output_set[200];
	sprintf(output_set, "%snorm_distribution.out", out_path);

	FILE *fp = fopen(output_set, "w");
	if (!fp) { printf("Could not create %s\n", output_set); return 1; }

	float num  = 0.5f / m;
	float step = 1.0f / m;
	for (int i = 0; i < m; ++i) {
		fprintf(fp, "%.1f\t%f\n", (num+step*i)*100.0f, freq[i]*100.0f/n);
	}
	fprintf(fp, "\n");
	fclose(fp);

	gettimeofday(&g_end_time, NULL);
	float runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
		(g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
	printf("Norm distribution: %.6f Seconds\n\n", runtime);

	return 0;
}

} // end namespace mips
