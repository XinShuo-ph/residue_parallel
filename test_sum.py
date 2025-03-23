#!/usr/bin/env python3

import time
import sympy
from sympy import symbols, together, fraction, expand, factor, simplify
import pickle
import os
import sys
from sym_utils import optimized_sum
import argparse

do_debug = True

def standard_sum(terms, debug=False):
    """The standard approach used in process_batch"""
    start_time = time.time()
    
    # Step 1: Sum terms and use together()
    together_start = time.time()
    initial_sum = sum(terms)
    if debug:
        print(f"Initial sum type: {'Mul' if initial_sum.is_Mul else 'Add' if initial_sum.is_Add else 'Other'}")
        if initial_sum.is_Add:
            print(f"Sum has {len(initial_sum.args)} terms")
    
    combined = together(initial_sum)
    together_end = time.time()
    
    if debug:
        print(f"After together() - Type: {'Mul' if combined.is_Mul else 'Add' if combined.is_Add else 'Other'}")
    
    # Step 2: Separate numerator and denominator
    fraction_start = time.time()
    num, den = fraction(combined)
    fraction_end = time.time()
    
    if debug:
        print(f"Fraction separation:")
        print(f"  - Numerator type: {'Mul' if num.is_Mul else 'Add' if num.is_Add else 'Other'}")
        print(f"  - Denominator type: {'Mul' if den.is_Mul else 'Add' if den.is_Add else 'Other'}")
        if den.is_Mul:
            print(f"  - Denominator has {len(den.args)} factors")
            # Show some example factors from denominator
            for i, myfac in enumerate(list(den.args)[:3]):
                print(f"  - Factor {i}: {myfac}")
    
    # Step 3: Expand the numerator
    expand_start = time.time()
    expanded_num = expand(num)
    expand_end = time.time()
    
    if debug:
        print(f"After expand() - Numerator:")
        print(f"  - Type: {'Mul' if expanded_num.is_Mul else 'Add' if expanded_num.is_Add else 'Other'}")
        if expanded_num.is_Add:
            print(f"  - Expanded numerator has {len(expanded_num.args)} terms")
    
    # Step 4: Factor the result
    factor_start = time.time()
    result = factor(expanded_num / den)
    factor_end = time.time()
    
    if debug:
        print(f"After factor() - Result:")
        print(f"  - Type: {'Mul' if result.is_Mul else 'Add' if result.is_Add else 'Other'}")
        if result.is_Mul:
            try:
                print(f"  - Result has {len(result.args)} factors")
                # Show some example factors from result
                for i, myfactor in enumerate(list(result.args)[:5]):
                    print(f"  - Factor {i}: {myfactor}")
            except Exception as e:
                print(f"  - Error listing factors: {e}")
        # Test if the denominator was eliminated during factorization
        try:
            test_num, test_den = fraction(result)
            if test_den != 1:
                if test_den.is_Mul:
                    print(f"  - Final denominator has {len(test_den.args)} factors")
                else:
                    print(f"  - Final denominator: {test_den}")
        except Exception as e:
            print(f"  - Error analyzing final fraction: {e}")
    
    end_time = time.time()
    
    times = {
        'together': together_end - together_start,
        'fraction': fraction_end - fraction_start,
        'expand': expand_end - expand_start,
        'factor': factor_end - factor_start,
        'total': end_time - start_time
    }
    
    return result, times
def main():
    # Check if file path is provided
    if len(sys.argv) < 2:
        print("Usage: python test_sum.py <path_to_f_current_pickle> [num_terms]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist")
        sys.exit(1)
    
    # Optional: number of terms to use in the test
    num_terms_to_use = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
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
    
    # Determine how many terms to use for testing
    if num_terms_to_use is None:
        # Default: use all terms if less than 10, otherwise use 5
        if num_terms <= 10:
            terms_to_use = list(f_current.args)
            print(f"Using all {num_terms} terms for testing")
        else:
            terms_to_use = list(f_current.args[:5])
            print(f"Using first 5 terms out of {num_terms} for testing")
    else:
        # Use specified number of terms
        if num_terms_to_use > num_terms:
            print(f"Requested {num_terms_to_use} terms, but only {num_terms} available")
            num_terms_to_use = num_terms
        
        terms_to_use = list(f_current.args[:num_terms_to_use])
        print(f"Using {num_terms_to_use} terms for testing")
    
    # Print term info
    for i, term in enumerate(terms_to_use):
        term_type = "Mul" if term.is_Mul else "Add" if term.is_Add else "Other"
        if term.is_Mul:
            num_factors = len(term.args)
            num_add_factors = sum(1 for arg in term.args if arg.is_Add)
            print(f"Term {i}: {term_type} with {num_factors} factors ({num_add_factors} Add factors)")
        elif term.is_Add:
            print(f"Term {i}: {term_type} with {len(term.args)} terms")
        else:
            print(f"Term {i}: {term_type}")
    
    # Test standard approach
    print("\n=== Testing standard approach ===")
    try:
        standard_start = time.time()
        result_standard, standard_times = standard_sum(terms_to_use, debug=do_debug)
        standard_end = time.time()
        
        print(f"Standard approach total time: {standard_end - standard_start:.4f}s")
        print(f"  - together(): {standard_times['together']:.4f}s")
        print(f"  - fraction(): {standard_times['fraction']:.4f}s")
        print(f"  - expand(): {standard_times['expand']:.4f}s")
        print(f"  - factor(): {standard_times['factor']:.4f}s")
        standard_succeeded = True
    except Exception as e:
        print(f"Standard approach failed with error: {e}")
        standard_succeeded = False
    
    # Test optimized approach
    print("\n=== Testing optimized_sum approach ===")
    try:
        optimized_start = time.time()
        result_optimized = optimized_sum(terms_to_use, debug=do_debug)
        optimized_end = time.time()
        print(f"Optimized approach total time: {optimized_end - optimized_start:.4f}s")
        optimized_succeeded = True
    except Exception as e:
        print(f"Optimized approach failed with error: {e}")
        optimized_succeeded = False
    
    # Verify results are equal if both approaches succeeded
    if standard_succeeded and optimized_succeeded:
        print("\n=== Verifying results ===")
        verify_start = time.time()
        try:
            difference = factor(result_standard / result_optimized)
            equal = difference == 1
            verify_end = time.time()
            print(f"Results are equal: {equal}")
            if not equal:
                print(f"Ratio: {difference}")
                print(f"standard result: {result_standard}")
                print(f"optimized_sum result: {result_optimized}")
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
    
    # Try with different batch sizes
    if num_terms >= 10 and num_terms_to_use is None:
        batch_sizes = [3, 5, 8]
        
        print("\n=== Testing different batch sizes ===")
        for batch_size in batch_sizes:
            if batch_size >= num_terms:
                continue
                
            batch = list(f_current.args[:batch_size])
            print(f"\nTesting with batch size {batch_size}")
            
            # Optimized approach
            try:
                optimized_start = time.time()
                result_batch = optimized_sum(batch, debug=do_debug)
                optimized_end = time.time()
                print(f"Optimized approach with {batch_size} terms: {optimized_end - optimized_start:.4f}s")
            except Exception as e:
                print(f"Optimized approach with {batch_size} terms failed: {e}")

    # Test incremental addition if there are enough terms
    if num_terms >= 5 and num_terms_to_use is None:
        print("\n=== Testing incremental addition ===")
        
        # Standard incremental approach (add terms one by one with standard method)
        try:
            standard_start = time.time()
            result = f_current.args[0]
            for i in range(1, 5):
                # Use standard_add logic directly
                combined = together(result + f_current.args[i])
                num, den = fraction(combined)
                expanded_num = expand(num)
                result = factor(expanded_num / den)
            standard_end = time.time()
            print(f"Standard incremental addition (5 terms): {standard_end - standard_start:.4f}s")
        except Exception as e:
            print(f"Standard incremental addition failed: {e}")
        
        # Optimized incremental approach (using optimized_add)
        try:
            optimized_start = time.time()
            result = f_current.args[0]
            for i in range(1, 5):
                result = optimized_add(result, f_current.args[i])
            optimized_end = time.time()
            print(f"Optimized incremental addition (5 terms): {optimized_end - optimized_start:.4f}s")
        except Exception as e:
            print(f"Optimized incremental addition failed: {e}")
        
        # Compare with one-shot optimized_sum
        try:
            oneshot_start = time.time()
            result = optimized_sum(list(f_current.args[:5]), debug=do_debug)
            oneshot_end = time.time()
            print(f"One-shot optimized_sum (5 terms): {oneshot_end - oneshot_start:.4f}s")
        except Exception as e:
            print(f"One-shot optimized_sum failed: {e}")

if __name__ == "__main__":
    main()