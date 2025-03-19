
import time
from sympy import symbols, collect, expand, simplify, together, fraction, diff, Abs, factor_list, solve, factor, multiplicity, limit, factorial
from sympy.printing.mathematica import mathematica_code
import sympy
import concurrent.futures
import os

def optimized_add(expr1, expr2):
    """
    Optimized addition for expressions that are products of a long polynomial and other factors.
    """
    start_time = time.time()
    
    def find_components(expr):
        """Extract numerical coefficient, long polynomial, and other factors from an expression."""
        component_start = time.time()
        
        if not expr.is_Mul:
            return 1, expr, [], time.time() - component_start
        
        num_coeff = 1
        long_poly = None
        other_factors = []
        
        for arg in expr.args:
            if arg.is_number:
                num_coeff *= arg
            elif arg.is_Add and len(arg.args) > 20:  # Heuristic for "long polynomial"
                long_poly = arg
            else:
                other_factors.append(arg)
        
        if long_poly is None and other_factors:
            # If no long polynomial was found, check if any factor is a product that contains a long polynomial
            for i, factor in enumerate(other_factors):
                if factor.is_Mul:
                    for subfactor in factor.args:
                        if subfactor.is_Add and len(subfactor.args) > 20:
                            long_poly = subfactor
                            other_factors.pop(i)
                            other_factors.extend([f for f in factor.args if f != subfactor])
                            break
                    if long_poly:
                        break
            
        # If still no long polynomial, take the first Add term as the polynomial
        if long_poly is None:
            for i, factor in enumerate(other_factors):
                if factor.is_Add:
                    long_poly = factor
                    other_factors.pop(i)
                    break
            
        # If still no long polynomial, just use 1
        if long_poly is None:
            long_poly = 1
            
        return num_coeff, long_poly, other_factors, time.time() - component_start
    
    # Extract components from both expressions
    component_time_start = time.time()
    coeff1, poly1, factors1, time1 = find_components(expr1)
    coeff2, poly2, factors2, time2 = find_components(expr2)
    component_time = time.time() - component_time_start
    
    # Find common factors
    common_factor_time_start = time.time()
    common_simple_factors = []
    common_pow_factors = {}  # Base -> min power
    unique_factors1 = []
    unique_factors2 = []
    
    # Separate Pow factors from other factors for better comparison
    simple_factors1 = []
    pow_factors1 = {}  # Base -> power
    
    for f in factors1:
        if f.is_Pow:
            base, power = f.args
            if base in pow_factors1:
                pow_factors1[base] += power
            else:
                pow_factors1[base] = power
        else:
            simple_factors1.append(f)
    
    simple_factors2 = []
    pow_factors2 = {}  # Base -> power
    
    for f in factors2:
        if f.is_Pow:
            base, power = f.args
            if base in pow_factors2:
                pow_factors2[base] += power
            else:
                pow_factors2[base] = power
        else:
            simple_factors2.append(f)
    
    # Compare simple factors
    sort_time_start = time.time()
    simple_factors1.sort(key=lambda x: str(x))
    simple_factors2.sort(key=lambda x: str(x))
    sort_time = time.time() - sort_time_start
    
    # Find common simple factors
    common_simple_time_start = time.time()
    i, j = 0, 0
    while i < len(simple_factors1) and j < len(simple_factors2):
        f1, f2 = simple_factors1[i], simple_factors2[j]
        if f1 == f2:
            common_simple_factors.append(f1)
            i += 1
            j += 1
        elif str(f1) < str(f2):  # Simple string comparison as a heuristic
            unique_factors1.append(f1)
            i += 1
        else:
            unique_factors2.append(f2)
            j += 1
    
    # Add remaining simple factors
    unique_factors1.extend(simple_factors1[i:])
    unique_factors2.extend(simple_factors2[j:])
    common_simple_time = time.time() - common_simple_time_start
    
    # Compare power factors
    pow_time_start = time.time()
    for base in set(pow_factors1.keys()) | set(pow_factors2.keys()):
        power1 = pow_factors1.get(base, 0)
        power2 = pow_factors2.get(base, 0)
        
        # Find common power (minimum of the two powers)
        min_power = min(power1, power2)
        
        if min_power != 0:
            common_pow_factors[base] = min_power
        
        # Add unique parts
        if power1 > min_power:
            unique_factors1.append(base ** (power1 - min_power))
        if power2 > min_power:
            unique_factors2.append(base ** (power2 - min_power))
    pow_time = time.time() - pow_time_start
    
    common_factor_time = time.time() - common_factor_time_start
    
    # Calculate the sum of the unique parts
    term_calc_time_start = time.time()
    term1 = coeff1 * poly1
    if unique_factors1:
        term1 *= sympy.Mul(*unique_factors1)
        
    term2 = coeff2 * poly2
    if unique_factors2:
        term2 *= sympy.Mul(*unique_factors2)
    
    # Expand the sum
    expand_time_start = time.time()
    expanded_sum = sympy.expand(term1 + term2)
    expand_time = time.time() - expand_time_start
    term_calc_time = time.time() - term_calc_time_start
    
    # Multiply by common factors
    final_time_start = time.time()
    result = expanded_sum
    
    # Multiply by common simple factors
    if common_simple_factors:
        result *= sympy.Mul(*common_simple_factors)
    
    # Multiply by common power factors
    for base, power in common_pow_factors.items():
        result *= base ** power
    
    final_time = time.time() - final_time_start
    total_time = time.time() - start_time
    
    print(f"Optimized add timing breakdown (total: {total_time:.4f}s):")
    print(f"  - Find components: {component_time:.4f}s (expr1: {time1:.4f}s, expr2: {time2:.4f}s)")
    print(f"  - Common factor analysis: {common_factor_time:.4f}s")
    print(f"    - Sorting: {sort_time:.4f}s")
    print(f"    - Common simple factors: {common_simple_time:.4f}s")
    print(f"    - Power factors: {pow_time:.4f}s")
    print(f"  - Term calculation: {term_calc_time:.4f}s")
    print(f"    - Expansion: {expand_time:.4f}s")
    print(f"  - Final multiplication: {final_time:.4f}s")
    
    return result