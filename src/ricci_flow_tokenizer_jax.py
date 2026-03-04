"""
RICCI FLOW TOKENIZATION - GPU Accelerated Implementation
Revolutionary approach: Tokens emerge from geometric flow dynamics

Usage:
  python ricci_flow_tokenizer_jax.py --text alice.txt --vocab_size 1000

Requirements:
  pip install jax[cuda12] matplotlib numpy tqdm
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time
from tqdm import tqdm

print("="*80)
print("RICCI FLOW TOKENIZATION - GPU Edition")
print("="*80)
print()

# Check GPU availability
print("JAX Configuration:")
print(f"  Devices: {jax.devices()}")
print(f"  Default backend: {jax.default_backend()}")
print()

# =============================================================================
# CORE RICCI FLOW ENGINE (GPU ACCELERATED)
# =============================================================================

@jit
def score_function(state, target, eps=1e-10):
    """
    Score function u(θ): Gradient of log-likelihood
    Pulls state toward observed frequencies
    
    Args:
        state: Current probability distribution (vocab_size,)
        target: Target distribution from data (vocab_size,)
        eps: Small constant for numerical stability
    
    Returns:
        Force vector pointing toward target
    """
    return (target - state) / (state + eps)

@jit
def entropy_gradient(state, eps=1e-10):
    """
    Gradient of Shannon entropy: -log(p) - 1
    Pushes toward uniform distribution (maximum entropy)
    
    Args:
        state: Current probability distribution (vocab_size,)
        eps: Small constant for numerical stability
    
    Returns:
        Force vector toward maximum entropy
    """
    return -jnp.log(state + eps) - 1.0

@jit
def curvature_force(state, correlation, eps=1e-10):
    """
    Simplified Ricci flow force
    Pulls together symbols that co-occur frequently
    
    This approximates the Ricci curvature gradient using
    the bigram correlation structure
    
    Args:
        state: Current probability distribution (vocab_size,)
        correlation: Bigram correlation matrix (vocab_size, vocab_size)
        eps: Small constant for numerical stability
    
    Returns:
        Force vector encouraging symbol merging
    """
    # For each symbol, compute attraction to other symbols
    # weighted by their correlation
    merge_pull = jnp.zeros_like(state)
    
    # Vectorized: for each symbol i, sum over symbols j
    # Pull proportional to correlation[i,j] and probability difference
    for i in range(len(state)):
        # Symbols that follow symbol i
        following = correlation[i, :]
        # Pull toward merging with correlated symbols
        merge_pull = merge_pull.at[i].set(
            -jnp.sum(following * (state - state[i]))
        )
    
    return merge_pull

@jit
def inverse_fisher_metric(state, eps=1e-10):
    """
    Inverse Fisher Information Matrix (diagonal approximation)
    
    For categorical distribution: g^{-1}_ii = p_i(1-p_i)
    
    Args:
        state: Current probability distribution (vocab_size,)
        eps: Small constant for numerical stability
    
    Returns:
        Diagonal inverse Fisher metric (vocab_size,)
    """
    return state * (1.0 - state) + eps

@jit
def flow_step(state, target, correlation, lambda_data, lambda_entropy, 
              lambda_curve, dt, eps=1e-10):
    """
    Single time step of Ricci flow
    
    Evolution equation:
        ∂θ/∂τ = g^{-1} · [λ₁·u(θ) + λ₂·∇H(θ) + λ₃·∇R(θ)]
    
    Args:
        state: Current state (vocab_size,)
        target: Target distribution (vocab_size,)
        correlation: Bigram correlation matrix (vocab_size, vocab_size)
        lambda_data: Strength of data attraction
        lambda_entropy: Strength of entropy repulsion
        lambda_curve: Strength of curvature force
        dt: Time step size
        eps: Numerical stability constant
    
    Returns:
        Updated state after one flow step
    """
    # Compute three forces
    force_data = score_function(state, target, eps)
    force_ent = entropy_gradient(state, eps)
    force_curv = curvature_force(state, correlation, eps)
    
    # Combined force
    total_force = (lambda_data * force_data + 
                   lambda_entropy * force_ent + 
                   lambda_curve * force_curv)
    
    # Natural gradient (Fisher-Rao geometry)
    g_inv = inverse_fisher_metric(state, eps)
    natural_velocity = g_inv * total_force
    
    # Euler step
    new_state = state + dt * natural_velocity
    
    # Project back to probability simplex
    new_state = jnp.maximum(new_state, eps)
    new_state = new_state / jnp.sum(new_state)
    
    return new_state

@partial(jit, static_argnums=(3, 4, 5, 6, 7))
def run_flow(initial_state, target, correlation, 
             T_max, dt, lambda_data, lambda_entropy, lambda_curve):
    """
    Run full Ricci flow evolution (GPU optimized)
    
    Args:
        initial_state: Starting distribution
        target: Target distribution from data
        correlation: Bigram correlation matrix
        T_max: Number of time steps
        dt: Time step size
        lambda_data, lambda_entropy, lambda_curve: Force strengths
    
    Returns:
        history: State at each time step (T_max+1, vocab_size)
    """
    def scan_fn(state, _):
        new_state = flow_step(state, target, correlation,
                             lambda_data, lambda_entropy, lambda_curve, dt)
        return new_state, new_state
    
    _, history = jax.lax.scan(scan_fn, initial_state, None, length=T_max)
    
    # Prepend initial state
    history = jnp.vstack([initial_state[None, :], history])
    
    return history

# =============================================================================
# DATA PROCESSING
# =============================================================================

def prepare_corpus(text, vocab_size=256):
    """
    Convert text to byte-level representation and compute statistics
    
    Args:
        text: Input text string
        vocab_size: Size of vocabulary (default 256 for bytes)
    
    Returns:
        bytes_array: Byte representation (numpy array)
        symbol_freq: Symbol frequencies (vocab_size,)
        correlation: Bigram correlation matrix (vocab_size, vocab_size)
    """
    # Convert to bytes
    bytes_array = np.array(list(text.encode('utf-8')), dtype=np.int32)
    
    # Clip to vocab_size (handle extended UTF-8)
    bytes_array = np.clip(bytes_array, 0, vocab_size - 1)
    
    # Count symbols
    symbol_counts = np.bincount(bytes_array, minlength=vocab_size)
    symbol_freq = symbol_counts / len(bytes_array)
    
    # Count bigrams
    correlation = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for i in range(len(bytes_array) - 1):
        correlation[bytes_array[i], bytes_array[i+1]] += 1
    
    # Normalize
    if correlation.sum() > 0:
        correlation /= correlation.sum()
    
    return bytes_array, symbol_freq, correlation

# =============================================================================
# VOCABULARY EXTRACTION
# =============================================================================

def compute_merge_affinity(history, correlation, window=10):
    """
    Compute which symbols want to merge based on flow trajectory
    
    High affinity = symbols pulled together during flow
    
    Args:
        history: Flow trajectory (T_max+1, vocab_size)
        correlation: Bigram correlation matrix
        window: Number of recent steps to average over
    
    Returns:
        affinity: Merge affinity matrix (vocab_size, vocab_size)
    """
    vocab_size = history.shape[1]
    affinity = np.zeros((vocab_size, vocab_size))
    
    # Average over last 'window' steps
    recent_states = np.array(history[-window:])
    
    for t in range(len(recent_states)):
        state = recent_states[t]
        
        for i in range(vocab_size):
            for j in range(i+1, vocab_size):
                # High correlation and similar probabilities → want to merge
                corr_score = correlation[i, j] + correlation[j, i]
                prob_similarity = np.exp(-10 * (state[i] - state[j])**2)
                affinity[i, j] += corr_score * prob_similarity
    
    # Normalize
    if affinity.max() > 0:
        affinity /= affinity.max()
    
    # Make symmetric
    affinity = affinity + affinity.T
    
    return affinity

def extract_vocabulary(affinity, bytes_array, top_k=100, threshold=0.01):
    """
    Extract recommended token vocabulary from merge affinity
    
    Args:
        affinity: Merge affinity matrix
        bytes_array: Original byte sequence
        top_k: Number of top tokens to extract
        threshold: Minimum affinity to consider
    
    Returns:
        tokens: List of (byte_sequence, affinity_score, frequency)
    """
    vocab_size = affinity.shape[0]
    
    # Find high-affinity pairs
    candidates = []
    for i in range(vocab_size):
        for j in range(i+1, vocab_size):
            if affinity[i, j] > threshold:
                # Count actual occurrences
                bigram = bytes([i, j])
                count = 0
                for k in range(len(bytes_array) - 1):
                    if bytes_array[k] == i and bytes_array[k+1] == j:
                        count += 1
                
                if count > 0:
                    candidates.append((bigram, affinity[i, j], count))
    
    # Sort by affinity
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    return candidates[:top_k]

# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_flow(history, symbol_freq, correlation, affinity, 
                   vocab_size=256, save_path='flow_results.png'):
    """
    Create comprehensive visualization of flow dynamics
    
    Args:
        history: Flow trajectory
        symbol_freq: Target symbol frequencies
        correlation: Bigram correlation matrix
        affinity: Merge affinity matrix
        vocab_size: Size of vocabulary
        save_path: Where to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Top symbol evolution
    ax = axes[0, 0]
    top_symbols = np.argsort(symbol_freq)[-10:]
    history_np = np.array(history)
    
    for idx in top_symbols:
        if symbol_freq[idx] > 0.001:  # Only plot significant symbols
            trajectory = history_np[:, idx]
            label = f"byte_{idx}" if idx < 128 else f"ext_{idx}"
            ax.plot(trajectory, label=label, alpha=0.7)
    
    ax.set_xlabel('Flow Time τ', fontsize=11)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title('Top 10 Symbol Probabilities Over Time', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Entropy evolution
    ax = axes[0, 1]
    entropies = []
    for state in history_np:
        H = -np.sum(state * np.log(state + 1e-10))
        entropies.append(H)
    
    ax.plot(entropies, linewidth=2, color='#2E86AB')
    ax.set_xlabel('Flow Time τ', fontsize=11)
    ax.set_ylabel('Entropy H(ρ)', fontsize=11)
    ax.set_title('Entropy Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: KL divergence
    ax = axes[0, 2]
    kl_divs = []
    for state in history_np:
        kl = np.sum(symbol_freq * np.log((symbol_freq + 1e-10) / (state + 1e-10)))
        kl_divs.append(kl)
    
    ax.plot(kl_divs, linewidth=2, color='#A23B72')
    ax.set_xlabel('Flow Time τ', fontsize=11)
    ax.set_ylabel('KL(Target || Flow)', fontsize=11)
    ax.set_title('Convergence to Data', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Correlation heatmap (zoomed)
    ax = axes[1, 0]
    # Show only printable ASCII range for clarity
    zoom_size = min(96, vocab_size)
    zoom_start = 32  # Space character
    corr_zoom = correlation[zoom_start:zoom_start+zoom_size, 
                           zoom_start:zoom_start+zoom_size]
    
    im = ax.imshow(corr_zoom, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.01)
    ax.set_xlabel('Following Byte', fontsize=11)
    ax.set_ylabel('Current Byte', fontsize=11)
    ax.set_title('Bigram Correlation (ASCII)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # Plot 5: Affinity heatmap (zoomed)
    ax = axes[1, 1]
    affinity_zoom = affinity[zoom_start:zoom_start+zoom_size,
                            zoom_start:zoom_start+zoom_size]
    
    im = ax.imshow(affinity_zoom, cmap='viridis', aspect='auto')
    ax.set_xlabel('Byte 2', fontsize=11)
    ax.set_ylabel('Byte 1', fontsize=11)
    ax.set_title('Merge Affinity (ASCII)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # Plot 6: Top tokens
    ax = axes[1, 2]
    top_pairs = []
    for i in range(zoom_start, min(zoom_start+zoom_size, vocab_size)):
        for j in range(i+1, min(zoom_start+zoom_size, vocab_size)):
            if affinity[i, j] > 0.01:
                top_pairs.append((i, j, affinity[i, j]))
    
    top_pairs.sort(key=lambda x: x[2], reverse=True)
    
    if top_pairs:
        top_10 = top_pairs[:10]
        labels = []
        values = []
        
        for i, j, aff in top_10:
            try:
                token = bytes([i, j]).decode('utf-8', errors='ignore')
                if not token:
                    token = f"{i},{j}"
            except:
                token = f"{i},{j}"
            labels.append(f"'{token}'")
            values.append(aff)
        
        ax.barh(range(len(values)), values, color='#2E86AB')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Merge Affinity', fontsize=11)
        ax.set_title('Top 10 Recommended Tokens', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    else:
        ax.text(0.5, 0.5, 'No strong tokens found', 
               ha='center', va='center', fontsize=12)
        ax.set_title('Top 10 Recommended Tokens', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {save_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main_simple_test():
    """
    Run on toy example to verify GPU acceleration
    """
    print("="*80)
    print("SIMPLE GPU TEST")
    print("="*80)
    print()
    
    # Toy corpus
    corpus = "ababcabcabc" * 100  # Repeat for more data
    
    print(f"Corpus: '{corpus[:50]}...' (length {len(corpus)})")
    print()
    
    # Prepare data
    print("Preparing data...")
    bytes_array, symbol_freq, correlation = prepare_corpus(corpus, vocab_size=256)
    
    # Convert to JAX arrays
    initial_state = jnp.array(symbol_freq)
    target = jnp.array(symbol_freq)
    correlation_jax = jnp.array(correlation)
    
    print(f"  Vocabulary size: {len(initial_state)}")
    print(f"  Non-zero symbols: {(symbol_freq > 0).sum()}")
    print()
    
    # Flow parameters
    T_max = 100
    dt = 0.05
    lambda_data = 1.0
    lambda_entropy = 0.1
    lambda_curve = 0.5
    
    # Run flow (GPU accelerated)
    print("Running Ricci flow on GPU...")
    start = time.time()
    
    history = run_flow(initial_state, target, correlation_jax,
                      T_max, dt, lambda_data, lambda_entropy, lambda_curve)
    
    # Force computation (JAX lazy evaluation)
    history = jax.block_until_ready(history)
    
    elapsed = time.time() - start
    print(f"✓ Completed {T_max} steps in {elapsed:.3f} seconds")
    print(f"  Speed: {T_max/elapsed:.1f} steps/sec")
    print()
    
    # Extract vocabulary
    print("Extracting vocabulary...")
    affinity = compute_merge_affinity(history, correlation)
    tokens = extract_vocabulary(affinity, bytes_array, top_k=20)
    
    print("\nTop recommended tokens:")
    for token_bytes, aff, count in tokens[:10]:
        try:
            token_str = token_bytes.decode('utf-8', errors='ignore')
        except:
            token_str = str(token_bytes)
        print(f"  '{token_str}': affinity={aff:.4f}, frequency={count}")
    
    # Visualize
    print("\nCreating visualization...")
    visualize_flow(history, symbol_freq, correlation, affinity,
                  save_path='/home/claude/gpu_test_results.png')
    
    print()
    print("="*80)
    print("GPU TEST COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main_simple_test()
