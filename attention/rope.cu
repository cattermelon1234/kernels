template<int HEAD_DIM>
__global__ void rope(
    float* q,
    float* k,
    int batch_size, 
    int num_heads, 
    int seq_len,
    int rotary_dim,
    float base
	  ) {
	  int batch = blockIdx.y
	  int head = blockIdx.x;

	  int pairs_per_token = rotary_dim / 2;
	  int total_pairs = seq_len * pairs_per_token;
	  int head_offset = ((batch * num_heads + head) * seq_len) * HEAD_DIM;

	  for (int idx = threadIdx.x; idx < total_pairs; idx += blockDim.x) {
	    int token_idx = idx / pairs_per_token;
	    int pair_idx = idx % pairs_per_token;
	    int dim_idx = pair_idx * 2;

	    float theta = powf(base, -static_cast<float>(dim_idx) / rotary_dim);
	    float cos_theta = cosf(theta * token_idx);
	    float sin_theta = sinf(theta * token_idx);
	    int offset = head_offset + token_idx * HEAD_DIM + dim_idx;

	    float q0 = q[offset];
	    float q1 = q[offset + 1];
	    float k0 = k[offset];
	    float k1 = k[offset + 1];

	    q[offset] = q0 * cos_theta - q1 * sin_theta;
	    q[offset + 1] = q0 * sin_theta + q1 * cos_theta;

	    k[offset] = k0 * cos_theta - k1 * sin_theta;
	    k[offset + 1] = k0 * sin_theta + k1 * cos_theta;
	  }
	}
