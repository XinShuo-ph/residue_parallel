#!/usr/bin/env python3

import time
import sympy
from sympy import symbols, together, fraction, expand, factor, simplify
import pickle
import os
import sys

# Import your optimized_add function
from sym_utils import optimized_add

def standard_add(term1, term2):
    """The standard approach used in process_batch"""
    combined_start = time.time()
    combined = together(term1 + term2)
    combined_end = time.time()
    
    fraction_start = time.time()
    num, den = fraction(combined)
    fraction_end = time.time()
    
    expand_start = time.time()
    expanded_num = expand(num)
    expand_end = time.time()
    
    factor_start = time.time()
    result = factor(expanded_num / den)
    factor_end = time.time()
    
    times = {
        'together': combined_end - combined_start,
        'fraction': fraction_end - fraction_start,
        'expand': expand_end - expand_start,
        'factor': factor_end - factor_start,
        'total': factor_end - combined_start
    }
    
    return result, times

def main():
    # Check if file path is provided
    if len(sys.argv) < 2:
        print("Usage: python test_optimized_add.py <path_to_f_current_pickle>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist")
        sys.exit(1)
    
    # Read f_current from file
    print(f"Loading f_current from {filepath}...")
    load_start = time.time()
    with open(filepath, 'rb') as f:
        f_current = pickle.load(f)
    load_end = time.time()
    print(f"Loaded in {load_end - load_start:.4f}s")
    
    # Get information about f_current
    if f_current.is_Add:
        num_terms = len(f_current.args)
        print(f"f_current is a sum with {num_terms} terms")
    else:
        print("f_current is not a sum, it's a single term")
        sys.exit(0)
    
    # Choose which terms to test
    try:
        # Choose the second and third terms if available
        if num_terms >= 3:
            term1_idx, term2_idx = 1, 2
            term1 = f_current.args[term1_idx]
            term2 = f_current.args[term2_idx]
            print(f"Using terms at indices {term1_idx} and {term2_idx}")
        else:
            print("Not enough terms in f_current, using first two terms")
            term1, term2 = f_current.args[0], f_current.args[1]
            term1_idx, term2_idx = 0, 1
    except Exception as e:
        print(f"Error selecting terms: {e}")
        sys.exit(1)
    
    print(f"\nTerm {term1_idx}: {term1}")
    print(f"Term {term2_idx}: {term2}")
    
    # Test standard approach
    print("\n=== Testing standard approach ===")
    standard_start = time.time()
    result_standard, standard_times = standard_add(term1, term2)
    standard_end = time.time()
    
    print(f"Standard approach total time: {standard_end - standard_start:.4f}s")
    print(f"  - together(): {standard_times['together']:.4f}s")
    print(f"  - fraction(): {standard_times['fraction']:.4f}s")
    print(f"  - expand(): {standard_times['expand']:.4f}s")
    print(f"  - factor(): {standard_times['factor']:.4f}s")
    
    # Test optimized approach
    print("\n=== Testing optimized_add approach ===")
    optimized_start = time.time()
    result_optimized = optimized_add(term1, term2)
    optimized_end = time.time()
    print(f"Optimized approach total time: {optimized_end - optimized_start:.4f}s")
    
    # Verify results are equal
    print("\n=== Verifying results ===")
    verify_start = time.time()
    try:
        difference = simplify(result_standard - result_optimized)
        equal = difference == 0
        verify_end = time.time()
        print(f"Results are equal: {equal}")
        if not equal:
            print(f"Difference: {difference}")
        print(f"Verification time: {verify_end - verify_start:.4f}s")
    except Exception as e:
        print(f"Error during verification: {e}")
    
    # Compare performance
    standard_time = standard_end - standard_start
    optimized_time = optimized_end - optimized_start
    speedup = standard_time / optimized_time if optimized_time > 0 else float('inf')
    
    print("\n=== Performance Summary ===")
    print(f"Standard approach: {standard_time:.4f}s")
    print(f"Optimized approach: {optimized_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # Try another test with different terms if available
    if num_terms >= 10:
        print("\n\n=== Additional Test with Different Terms ===")
        term3_idx, term4_idx = 5, 8  # Choose some different terms
        term3 = f_current.args[term3_idx]
        term4 = f_current.args[term4_idx]
        
        print(f"Term {term3_idx}: {term3}")
        print(f"Term {term4_idx}: {term4}")
        
        # Test standard approach
        print("\n=== Testing standard approach ===")
        standard_start = time.time()
        result_standard, standard_times = standard_add(term3, term4)
        standard_end = time.time()
        
        print(f"Standard approach total time: {standard_end - standard_start:.4f}s")
        print(f"  - together(): {standard_times['together']:.4f}s")
        print(f"  - fraction(): {standard_times['fraction']:.4f}s")
        print(f"  - expand(): {standard_times['expand']:.4f}s")
        print(f"  - factor(): {standard_times['factor']:.4f}s")
        
        # Test optimized approach
        print("\n=== Testing optimized_add approach ===")
        optimized_start = time.time()
        result_optimized = optimized_add(term3, term4)
        optimized_end = time.time()
        print(f"Optimized approach total time: {optimized_end - optimized_start:.4f}s")
        
        # Compare performance
        standard_time = standard_end - standard_start
        optimized_time = optimized_end - optimized_start
        speedup = standard_time / optimized_time if optimized_time > 0 else float('inf')
        
        print("\n=== Performance Summary (Second Test) ===")
        print(f"Standard approach: {standard_time:.4f}s")
        print(f"Optimized approach: {optimized_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()