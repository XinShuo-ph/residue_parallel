
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

def optimized_sum(expr_list, debug=False):
    """
    Optimized summation for a list of expressions that are products of a long polynomial and other factors.
    Follows the same logic as optimized_add but works on multiple expressions at once.
    
    Parameters:
    expr_list - List of expressions to sum
    debug - If True, print detailed debugging information
    
    Returns:
    Simplified sum of the expressions
    """
    if not expr_list:
        return 0
    
    if len(expr_list) == 1:
        return expr_list[0]
    
    start_time = time.time()
        
    def find_components(expr):
        """Extract numerical coefficient, long polynomial, and other factors from an expression."""
        component_start = time.time()
        
        if not expr.is_Mul:
            if debug:
                print(f"Expression is not a product, returning as-is: {expr}")
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
        
        if debug:
            print(f"Initial component extraction:")
            print(f"  - Numerical coefficient: {num_coeff}")
            print(f"  - Long polynomial found: {long_poly is not None}")
            print(f"  - Other factors count: {len(other_factors)}")
        
        if long_poly is None and other_factors:
            # If no long polynomial was found, check if any factor is a product that contains a long polynomial
            for i, fac in enumerate(other_factors):
                if fac.is_Mul:
                    for subfactor in fac.args:
                        if subfactor.is_Add and len(subfactor.args) > 20:
                            long_poly = subfactor
                            other_factors.pop(i)
                            other_factors.extend([f for f in fac.args if f != subfactor])
                            if debug:
                                print(f"  - Found long polynomial in nested Mul: {long_poly}")
                            break
                    if long_poly:
                        break
            
        # If still no long polynomial, take the first Add term as the polynomial
        if long_poly is None:
            for i, fac in enumerate(other_factors):
                if fac.is_Add:
                    long_poly = fac
                    other_factors.pop(i)
                    if debug:
                        print(f"  - Using first Add term as polynomial: {long_poly}")
                    break
            
        # If still no long polynomial, just use 1
        if long_poly is None:
            long_poly = 1
            if debug:
                print("  - No polynomial found, using 1")
            
        return num_coeff, long_poly, other_factors, time.time() - component_start

    # Extract components from all expressions
    component_time_start = time.time()
    components = []
    component_times = []
    
    print(f"Processing {len(expr_list)} terms")
    for i, expr in enumerate(expr_list):
        if debug:
            print(f"\nExtracting components for term {i}:")
        coeff, poly, factors, time_taken = find_components(expr)
        components.append((coeff, poly, factors))
        component_times.append(time_taken)
        if debug:
            print(f"Term {i} components:")
            print(f"  - Coefficient: {coeff}")
            print(f"  - Polynomial: {poly}")
            print(f"  - Factor count: {len(factors)}")
            # Print factor types for debugging
            factor_types = {
                "simple": [f for f in factors if not f.is_Pow],
                "power": [f for f in factors if f.is_Pow]
            }
            print(f"  - Simple factors: {len(factor_types['simple'])}")
            print(f"  - Power factors: {len(factor_types['power'])}")
            if i == 0 and len(factor_types['simple']) > 0:
                print(f"  - Example simple factor: {factor_types['simple'][0]}")
            if i == 0 and len(factor_types['power']) > 0:
                print(f"  - Example power factor: {factor_types['power'][0]}")
    
    component_time = time.time() - component_time_start
    print(f"Component extraction completed in {component_time:.4f}s")
    
    # Process factors to identify common ones
    factor_processing_start = time.time()
    
    # First, separate simple factors and power factors for each expression
    all_simple_factors = []
    all_pow_factors = []  # List of dictionaries: Base -> power
    
    for idx, (_, _, factors) in enumerate(components):
        simple_factors = []
        pow_factors = {}  # Base -> power
        
        for f in factors:
            if f.is_Pow:
                base, power = f.args
                if base in pow_factors:
                    pow_factors[base] += power
                else:
                    pow_factors[base] = power
            else:
                simple_factors.append(f)
        
        all_simple_factors.append(simple_factors)
        all_pow_factors.append(pow_factors)
        
        if debug:
            print(f"\nTerm {idx} factor separation:")
            print(f"  - Simple factors: {len(simple_factors)}")
            print(f"  - Power factor bases: {len(pow_factors)}")
            if len(simple_factors) > 0:
                print(f"  - First simple factor: {simple_factors[0]}")
            if len(pow_factors) > 0:
                base = list(pow_factors.keys())[0]
                print(f"  - Example power factor: {base}^{pow_factors[base]}")
   
    # Find common factors across all expressions
    common_factors_start = time.time()

    # First convert all simple factors to power factors with power 1
    unified_pow_factors = []
    for idx, (simple_factors, pow_factors) in enumerate(zip(all_simple_factors, all_pow_factors)):
        # Copy the power factors dictionary
        unified_dict = pow_factors.copy()
        
        # Add simple factors as power factors with power 1
        for sf in simple_factors:
            if sf in unified_dict:
                unified_dict[sf] += 1
            else:
                unified_dict[sf] = 1
        
        unified_pow_factors.append(unified_dict)
        
        if debug:
            print(f"\nTerm {idx} unified factors:")
            print(f"  - Total unique bases: {len(unified_dict)}")
            if unified_dict:
                sample_base = next(iter(unified_dict))
                print(f"  - Example: {sample_base}^{unified_dict[sample_base]}")

    # Get all unique bases across all expressions
    all_bases = set()
    for unified_dict in unified_pow_factors:
        all_bases.update(unified_dict.keys())

    if debug:
        print(f"\nFound {len(all_bases)} unique factor bases across all terms")
        for i, base in enumerate(list(all_bases)[:5]):
            print(f"  - Base {i}: {base}")

    # Find common factors (bases that appear in all expressions with their minimum powers)
    common_factors = {}

    for base in all_bases:
        # Extract power from each expression or default to 0 if not present
        powers = [unified_dict.get(base, 0) for unified_dict in unified_pow_factors]
        min_power = min(powers)
        
        # Only include as common if present in all expressions (min_power != 0)
        if min_power != 0:
            common_factors[base] = min_power
            if debug:
                print(f"  - Common factor: {base}^{min_power} (powers: {powers})")
        elif debug:
            print(f"  - Not common: {base} (powers: {powers})")

    if debug:
        print(f"\nFound {len(common_factors)} common factor bases")
        for i, (base, power) in enumerate(list(common_factors.items())[:5]):
            print(f"  - Common factor {i}: {base}^{power}")

    # Now extract the unique factors for each expression after removing common factors
    unique_factors_list = []

    for idx, unified_dict in enumerate(unified_pow_factors):
        unique_factors = []
        
        if debug:
            print(f"\nProcessing unique factors for term {idx}:")
            print(f"  - unified factors: {unified_dict}")
        
        # Process all bases
        for base, power in unified_dict.items():
            common_power = common_factors.get(base, 0)
            if power > common_power:
                # This expression has a higher power of this base, add the difference
                unique_factors.append(base ** (power - common_power))
        
        # Also check if any common base is missing in this expression
        for base, common_power in common_factors.items():
            if base not in unified_dict:
                # This expression is missing this factor, add with negative power
                unique_factors.append(base ** (-common_power))
        
        unique_factors_list.append(unique_factors)
        
        if debug:
            print(f"Term {idx} unique factors:")
            print(f"  - Unique factors: {len(unique_factors)}")
            if len(unique_factors) > 0:
                print(f"  - Example unique factor: {unique_factors[0]}")

    common_factors_time = time.time() - common_factors_start
    factor_processing_time = time.time() - factor_processing_start

    print(f"Factor processing completed in {factor_processing_time:.4f}s")
    print(f"Found {len(common_factors)} common factors")

    # The rest of the function continues as before, but using common_factors instead of 
    # common_simple_factors and common_pow_factors

    # Calculate terms with unique factors
    term_calc_start = time.time()
    terms = []

    for i, ((coeff, poly, _), unique_factors) in enumerate(zip(components, unique_factors_list)):
        term = coeff * poly
        if unique_factors:
            term *= sympy.Mul(*unique_factors)
        terms.append(term)
        
        if debug:
            print(f"\nTerm {i} after applying unique factors:")
            print(f"  - Original coefficient: {coeff}")
            print(f"  - Original polynomial: {poly}")
            print(f"  - Unique factors: {len(unique_factors)}")
            print(f"  - Resulting term type: {'Mul' if term.is_Mul else 'Add' if term.is_Add else 'Other'}")

    # Calculate expanded sum of terms
    expand_start = time.time()
    expanded_sum = sympy.expand(sum(terms))
    expand_time = time.time() - expand_start
    term_calc_time = time.time() - term_calc_start

    print(f"Term calculation completed in {term_calc_time:.4f}s (expand: {expand_time:.4f}s)")

    if debug:
        print("\nExpanded sum information:")
        print(f"  - Type: {'Mul' if expanded_sum.is_Mul else 'Add' if expanded_sum.is_Add else 'Other'}")
        if expanded_sum.is_Add:
            print(f"  - Number of terms: {len(expanded_sum.args)}")

    # Multiply by common factors
    final_time_start = time.time()
    result = expanded_sum

    # Multiply by common factors
    if common_factors:
        if debug:
            print(f"\nMultiplying by {len(common_factors)} common factors")
            for i, (base, power) in enumerate(list(common_factors.items())[:3]):
                print(f"  - Factor {i}: {base}^{power}")
        
        for base, power in common_factors.items():
            result *= base ** power
    
    # Debug final result
    if debug:
        print("\nFinal result information:")
        print(f"  - Type: {'Mul' if result.is_Mul else 'Add' if result.is_Add else 'Other'}")
        if result.is_Add:
            print(f"  - Number of terms: {len(result.args)}")
        # Test factorization to see actual factors
        try:
            test_factored = factor(result)
            print("  - Factorized form:")
            if test_factored.is_Mul:
                print(f"    - Number of factors: {len(test_factored.args)}")
                print(f"    - First few factors: {', '.join(str(f) for f in list(test_factored.args)[:3])}")
            else:
                print(f"    - Not a product after factorization: {test_factored}")
        except Exception as e:
            print(f"  - Error during test factorization: {e}")
    
    final_time = time.time() - final_time_start
    total_time = time.time() - start_time
    
    print(f"Optimized sum timing breakdown (total: {total_time:.4f}s):")
    print(f"  - Find components: {component_time:.4f}s")
    print(f"  - Factor processing: {factor_processing_time:.4f}s")
    print(f"    - Common factors: {common_factors_time:.4f}s")
    print(f"  - Term calculation: {term_calc_time:.4f}s")
    print(f"    - Expansion: {expand_time:.4f}s")
    print(f"  - Final multiplication: {final_time:.4f}s")
    
    return result