"""
Parallel Utilities for RL Experiments
======================================

Provides utilities for:
1. Automatic device selection (CUDA/MPS/CPU)
2. Parallel execution of experiments
3. Progress tracking
"""

import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Dict, Any
import psutil
import time
from tqdm import tqdm


# ============================================================
# Device Selection
# ============================================================

def get_best_device(verbose=False):
    """
    Automatically detect and return the best available device.
    
    Priority:
    1. CUDA (NVIDIA GPU) - best for large batches
    2. MPS (Apple Silicon) - good for M1/M2/M3 Macs
    3. CPU - fallback
    
    Returns:
        str: 'cuda', 'mps', or 'cpu'
    """
    
    # Check CUDA
    if torch.cuda.is_available():
        device = 'cuda'
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ Using CUDA: {gpu_name} ({gpu_mem:.1f} GB)")
    
    # Check MPS (Apple Silicon)
    elif torch.backends.mps.is_available():
        device = 'mps'
        if verbose:
            print(f"✓ Using MPS: Apple Silicon GPU")
    
    # Fallback to CPU
    else:
        device = 'cpu'
        if verbose:
            cpu_count = mp.cpu_count()
            print(f"✓ Using CPU: {cpu_count} cores available")
    
    return device


def benchmark_devices(dummy_forward_fn, num_iterations=100, verbose=True):
    """
    Benchmark different devices to find the fastest.
    
    Args:
        dummy_forward_fn: A function that performs a typical forward pass
        num_iterations: Number of iterations to average
        verbose: Print results
    
    Returns:
        str: Best device name
    """
    devices_to_test = []
    
    if torch.cuda.is_available():
        devices_to_test.append('cuda')
    if torch.backends.mps.is_available():
        devices_to_test.append('mps')
    devices_to_test.append('cpu')
    
    results = {}
    
    for device in devices_to_test:
        # Warmup
        for _ in range(10):
            dummy_forward_fn(device)
        
        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            dummy_forward_fn(device)
        elapsed = time.time() - start
        
        results[device] = elapsed
        
        if verbose:
            print(f"  {device.upper():6s}: {elapsed:.3f}s ({num_iterations} iterations)")
    
    best_device = min(results, key=results.get)
    
    if verbose:
        speedup = results['cpu'] / results[best_device]
        print(f"\n  → Best: {best_device.upper()} ({speedup:.2f}x faster than CPU)")
    
    return best_device


# ============================================================
# Parallel Execution
# ============================================================

def get_optimal_workers(device='cpu', reserve_cores=1):
    """
    Determine optimal number of parallel workers.
    
    Args:
        device: 'cpu', 'cuda', or 'mps'
        reserve_cores: Number of cores to reserve for system
    
    Returns:
        int: Recommended number of workers
    """
    
    if device == 'cuda':
        # For CUDA, can run multiple experiments sequentially on GPU
        # But parallel CPU processes for environment simulation
        return max(1, mp.cpu_count() - reserve_cores)
    
    elif device == 'mps':
        # MPS doesn't support multi-process well, use fewer workers
        return max(1, min(4, mp.cpu_count() - reserve_cores))
    
    else:  # CPU
        # Use all available cores minus reserve
        available_cores = mp.cpu_count()
        return max(1, available_cores - reserve_cores)


def run_experiments_parallel(
    experiment_configs: List[Dict[str, Any]],
    experiment_fn: Callable,
    max_workers: int = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Run multiple experiments in parallel.
    
    Args:
        experiment_configs: List of config dicts for each experiment
        experiment_fn: Function that takes a config and returns results
        max_workers: Maximum number of parallel workers (None = auto)
        verbose: Show progress bar
    
    Returns:
        List of results in the same order as configs
    """
    
    if max_workers is None:
        max_workers = get_optimal_workers()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Running {len(experiment_configs)} experiments with {max_workers} workers")
        print(f"{'='*70}\n")
    
    results = [None] * len(experiment_configs)
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_idx = {
            executor.submit(experiment_fn, config): idx
            for idx, config in enumerate(experiment_configs)
        }
        
        # Track completion with progress bar
        if verbose:
            pbar = tqdm(total=len(experiment_configs), desc="Experiments")
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
                
                if verbose:
                    # Update progress bar with info
                    config = experiment_configs[idx]
                    desc = f"{config.get('agent_type', 'agent')} β={config.get('beta', 'N/A'):.2f}"
                    pbar.set_postfix_str(desc)
                    pbar.update(1)
                    
            except Exception as e:
                print(f"\n✗ Experiment {idx} failed: {e}")
                results[idx] = {'error': str(e), 'config': experiment_configs[idx]}
                
                if verbose:
                    pbar.update(1)
        
        if verbose:
            pbar.close()
    
    return results


def create_experiment_configs(
    agent_types: List[str],
    beta_values: List[float],
    num_runs: int,
    base_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Create all experiment configurations for parallel execution.
    
    Args:
        agent_types: List of agent types (e.g., ['leaky', 'rleaky'])
        beta_values: List of beta values to test
        num_runs: Number of runs per configuration
        base_config: Base configuration dict
    
    Returns:
        List of experiment configs
    """
    
    configs = []
    
    for arch_type in agent_types:
        for beta in beta_values:
            for run_id in range(num_runs):
                config = base_config.copy()
                config.update({
                    'agent_type': f'{arch_type}_fixed',
                    'beta': beta,
                    'run_id': run_id,
                    'arch_name': arch_type,  # For display
                })
                configs.append(config)
    
    return configs


# ============================================================
# Resource Monitoring
# ============================================================

def get_system_info(verbose=True):
    """
    Get and optionally print system information.
    
    Returns:
        dict: System information
    """
    
    info = {
        'cpu_count': mp.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_total': psutil.virtual_memory().total / 1e9,
        'memory_available': psutil.virtual_memory().available / 1e9,
        'memory_percent': psutil.virtual_memory().percent,
    }
    
    # Check GPU
    if torch.cuda.is_available():
        info['gpu_available'] = True
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    elif torch.backends.mps.is_available():
        info['gpu_available'] = True
        info['gpu_name'] = 'Apple Silicon MPS'
        info['gpu_memory'] = 'Shared with system'
    else:
        info['gpu_available'] = False
    
    if verbose:
        print("\n" + "="*70)
        print("SYSTEM INFORMATION")
        print("="*70)
        print(f"CPU Cores: {info['cpu_count']}")
        print(f"CPU Usage: {info['cpu_percent']:.1f}%")
        print(f"RAM: {info['memory_available']:.1f} GB / {info['memory_total']:.1f} GB available")
        print(f"RAM Usage: {info['memory_percent']:.1f}%")
        
        if info['gpu_available']:
            print(f"\nGPU: {info['gpu_name']}")
            if isinstance(info['gpu_memory'], float):
                print(f"VRAM: {info['gpu_memory']:.1f} GB")
        else:
            print(f"\nGPU: Not available")
        
        print("="*70 + "\n")
    
    return info


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    """
    Example of how to use these utilities
    """
    
    # 1. Get system info
    get_system_info(verbose=True)
    
    # 2. Find best device
    device = get_best_device(verbose=True)
    print(f"\nRecommended device: {device}")
    
    # 3. Get optimal workers
    workers = get_optimal_workers(device)
    print(f"Recommended workers: {workers}\n")
    
    # 4. Example experiment function
    def dummy_experiment(config):
        """Dummy experiment for testing"""
        time.sleep(2)  # Simulate 2-second experiment
        return {
            'agent_type': config['agent_type'],
            'beta': config['beta'],
            'run_id': config['run_id'],
            'result': 'success'
        }
    
    # 5. Create configs and run in parallel
    configs = create_experiment_configs(
        agent_types=['leaky', 'rleaky'],
        beta_values=[0.3, 0.7],
        num_runs=2,
        base_config={'device': device}
    )
    
    print(f"Total experiments: {len(configs)}")
    print("Running in parallel...\n")
    
    results = run_experiments_parallel(
        experiment_configs=configs,
        experiment_fn=dummy_experiment,
        max_workers=workers,
        verbose=True
    )
    
    print(f"\nCompleted {len([r for r in results if r is not None])} experiments")
