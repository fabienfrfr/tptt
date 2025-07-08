    @staticmethod
    def chunk_delta_product_forward(
        query,
        key,
        value,
        beta_gate,
        chunk_size,
        n=1,
        trick="derivative",
        linear=True,
        initial_state=None,
    ):
        """
        DeltaProduct implementation https://arxiv.org/abs/2502.10297
        Chunkwise parallel implementation https://arxiv.org/abs/2406.06484
        """

        def sequential_delta_product_scan(
            q_chunks,
            W,
            U,
            n_orders,
            linear_activation,
            current_chunk_size,
            initial_recurrent_state,
        ):
            """
            For each chunk, processes chunk_size * n_orders steps (virtual tokens) in order.
            This function implements the per-token Householder state updates.
            """
            # chunk_n_total = current_chunk_size * n_orders
            B, H, num_chunks_inner, chunk_n_total, D = q_chunks.shape

            output_inner = torch.empty_like(q_chunks)
            # Ensure initial_recurrent_state is H_{last_token_of_prev_chunk, n-1} ([B, H, D, D])
            # If initial_recurrent_state comes as [B, H, n_orders, D, D], select the last order.
            if (
                initial_recurrent_state.dim() == 5
                and initial_recurrent_state.shape[2] == n_orders
            ):
                # Use the highest order state of the last token - Shape: [B, H, D, D]
                h_0_base = initial_recurrent_state[:, :, -1, :, :].clone()
            elif (
                initial_recurrent_state.dim() == 4
            ):  # Assuming it's already [B, H, D, D]
                h_0_base = initial_recurrent_state.clone()
            else:
                raise ValueError(
                    f"Unexpected initial_recurrent_state shape: {initial_recurrent_state.shape}. Expected [B,H,D,D] or [B,H,n_orders,D,D]."
                )

            # This will hold the 'n_orders' states for the *last token* of the current chunk,
            # for propagation to the next chunk. Its shape will be [B, H, n_orders, D, D].
            propagated_state_for_next_chunk_boundary = torch.zeros(
                B, H, n_orders, D, D, device=q_chunks.device, dtype=torch.float32
            )

            for chunk_idx_inner in range(num_chunks_inner):
                # Extract chunk-specific parameters [B, H, chunk_n_total, D]
                q_chunk_params = q_chunks[:, :, chunk_idx_inner]
                w_chunk_params = W[:, :, chunk_idx_inner]
                u_chunk_params = U[:, :, chunk_idx_inner]

                o_intra_current_chunk = torch.zeros(
                    B,
                    H,
                    chunk_n_total,
                    D,
                    device=q_chunk_params.device,
                    dtype=torch.float32,
                )
                o_inter_current_chunk = torch.zeros(
                    B,
                    H,
                    chunk_n_total,
                    D,
                    device=q_chunk_params.device,
                    dtype=torch.float32,
                )

                # Initialize per-token Householder state for this chunk. (start with the same H_{t-1, n-1})
                current_accumulated_state_per_token = (
                    h_0_base.unsqueeze(2)
                    .expand(-1, -1, current_chunk_size, -1, -1)
                    .clone()
                )  # Shape: [B, H, current_chunk_size, D, D]
                # Looping over Householder orders (j in H_{i,j})
                for step in range(n_orders):
                    # Select the parameters for this step/order for all tokens in the chunk.
                    idx_virtual_tokens = (
                        torch.arange(current_chunk_size, device=q_chunk_params.device)
                        * n_orders
                        + step
                    )
                    # [B, H, current_chunk_size, D]
                    q_s = q_chunk_params[:, :, idx_virtual_tokens, :]
                    # k_s = k_chunk_params[:, :, idx_virtual_tokens, :]
                    w_s = w_chunk_params[:, :, idx_virtual_tokens, :]
                    u_s = u_chunk_params[:, :, idx_virtual_tokens, :]

                    # state_input_for_this_step is H_{i,j-1} for all tokens i.
                    state_input_for_this_step = current_accumulated_state_per_token  # [B, H, current_chunk_size, D, D]

                    # Calculate u_val (term: v^T - k^T H_old) per token
                    k_trans_h_old = torch.einsum(
                        "bhcd,bhcdd->bhcd", w_s, state_input_for_this_step
                    )
                    u_val = u_s - k_trans_h_old  # [B, H, current_chunk_size, D]

                    # Calculate o_inter (q_i @ H_{i,j-1}) per token
                    o_inter_current_chunk[:, :, idx_virtual_tokens, :] = torch.einsum(
                        "bhcd,bhcdd->bhcd", q_s, state_input_for_this_step
                    ).to(dtype=torch.float32)

                    # Calculate DeltaProduct output formula (causal attention-like term): q_s * u_val
                    o_intra_current_chunk[:, :, idx_virtual_tokens, :] = torch.einsum(
                        "bhcd,bhcd->bhcd", q_s, u_val
                    ).to(dtype=torch.float32)

                    # Update the Householder state (H_{i,j} = H_{i,j-1} + k_{i,j} (v_{i,j}^T - k_{i,j}^T H_{i,j-1}))
                    outer_product_term = torch.matmul(
                        w_s.unsqueeze(-1), u_val.unsqueeze(-2)
                    )  # [B, H, current_chunk_size, D, D]

                    new_state_i_per_token = (
                        state_input_for_this_step + outer_product_term
                    )  # [B, H, current_chunk_size, D, D]
                    new_state_i_per_token = ensure_stability(
                        new_state_i_per_token, min_val=-1e4, max_val=1e4
                    )

                    # Update the per-token accumulated state for the next step/order.
                    current_accumulated_state_per_token = new_state_i_per_token.to(
                        dtype=torch.float32
                    )

                    # Store the 'n_orders' states for the *last token* of the chunk for boundary propagation.
                    # This will be the new 'initial_recurrent_state' for the next chunk.
                    # We store H_{last_token, step} in propagated_state_for_next_chunk_boundary[:, :, step]
                    propagated_state_for_next_chunk_boundary[:, :, step] = (
                        current_accumulated_state_per_token[:, :, -1, :, :].to(
                            dtype=torch.float32
                        )
                    )

                # Combine intra and inter outputs for the current chunk.
                output_inner[:, :, chunk_idx_inner] = (
                    o_intra_current_chunk + o_inter_current_chunk
                )

                # The h_0_base for the next chunk is the highest order state of the last token of the current chunk
                h_0_base = propagated_state_for_next_chunk_boundary[
                    :, :, -1, :, :
                ].clone()

            # Apply non-linear activation if required (more RNN-like) to the state propagated between chunks
            final_propagated_state_for_all_orders = (
                propagated_state_for_next_chunk_boundary
                if linear_activation
                else F.gelu(
                    propagated_state_for_next_chunk_boundary, approximate="tanh"
                ).to(dtype=torch.float32)
            )  # pylint: disable=not-callable

            return output_inner, final_propagated_state_for_all_orders

        # --- Main chunk_delta_product_forward logic ---

        batch_size, num_heads, seq_len, head_dim = query.shape
        chunk_size = get_valid_chunk_size(seq_len, chunk_size)
        num_chunks = seq_len // chunk_size

        # Prepare product scan (trick to simulate multihead): [B, H, seq_len*n, D]
        query_n = query if n == 1 else expand_virtual_tokens(query, n, trick)
        key_n = key if n == 1 else expand_virtual_tokens(key, n, trick)
        value_n = value if n == 1 else expand_virtual_tokens(value, n, trick)
        beta_n = beta_gate if n == 1 else expand_virtual_tokens(beta_gate, n, trick)

        # Chunk input tensors to [B, H, num_chunks, chunk_size*n, D]
        q_chunks = chunk_sequence(query_n, num_chunks, chunk_size * n)
        k_chunks = chunk_sequence(key_n, num_chunks, chunk_size * n)
        v_chunks = chunk_sequence(value_n, num_chunks, chunk_size * n)
        beta_chunks = chunk_sequence(beta_n, num_chunks, chunk_size * n)

        # Gated keys/values: [B, H, C, chunk_size*n, D]
        k_beta = k_chunks * beta_chunks
        v_beta = v_chunks * beta_chunks

        # Build strictly lower-triangular T: [B, H, C, chunk_size*n, chunk_size*n]
        # This T matrix still handles virtual tokens within a chunk's internal processing.
        T = -(k_beta @ k_chunks.transpose(-2, -1)).tril(-1)
        T = ensure_stability(T, min_val=-1e4, max_val=1e4)

        # Invert (I - T): [B, H, C, chunk_size*n, chunk_size*n]
        inv_T = invert_nchunked_lower_triangular_matrix(T)

        # Compute W and U: [B, H, C, chunk_size*n, D]
        W = ensure_stability(torch.matmul(inv_T, k_beta), min_val=-1e4, max_val=1e4)
        U = ensure_stability(torch.matmul(inv_T, v_beta), min_val=-1e4, max_val=1e4)

        # This initial_state represents the 'n' states of the *last token* from the *previous sequence/chunk*.
        state_shape = (batch_size, num_heads, n, head_dim, head_dim)
        if initial_state is not None and initial_state.shape == state_shape:
            state = initial_state.to(device=query.device, dtype=torch.float32)
        else:
            # Initialize with a small value for numerical stability
            state = torch.full(
                state_shape,
                fill_value=1e-6,
                device=query.device,
                dtype=torch.float32,
            )

        # Sequential scan over chunks using the DeltaProduct rule
        # The inner sequential_delta_product_scan now correctly processes states per token.
        output, final_state = sequential_delta_product_scan(
            q_chunks.to(dtype=torch.float32),
            W.to(dtype=torch.float32),
            U.to(dtype=torch.float32),
            n,  # Passed as n_orders to inner function
            linear,  # Passed as linear_activation to inner function
            chunk_size,  # Passed as current_chunk_size to inner function
            state.to(dtype=torch.float32),  # initial_recurrent_state for inner function
        )

        # Restore output shape to [batch, num_heads, seq_len, head_dim]
        # We only care about the highest order (n-1) output for each physical token.
        idx_last_order = torch.arange(chunk_size, device=output.device) * n + (n - 1)
        output = output[:, :, :, idx_last_order, :]  # [B, H, num_chunks, chunk_size, D]
        output = output.reshape(batch_size, num_heads, seq_len, head_dim)

        return output.to(dtype=torch.float32), final_state.to(dtype=torch.float32)