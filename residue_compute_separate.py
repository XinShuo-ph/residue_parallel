#!/usr/bin/env python3

import time
from sympy import symbols, collect, expand, simplify, together, fraction, diff, Abs, factor_list, solve, factor, multiplicity, limit, factorial
from sympy.printing.mathematica import mathematica_code
import sympy
import concurrent.futures
import os
import argparse
from sym_utils import optimized_sum, optimized_add

import psutil  

def get_memory_usage():
    """Return memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert bytes to MB

def log_memory(label):
    """Log current memory usage with a label"""
    mem_mb = get_memory_usage()
    if mem_mb > 100:  # Only log if >100MB
        print(f"Memory usage ({label}): {mem_mb:.2f} MB")

sympy.init_printing()  # for pretty printing if using an interactive environment

###############################################################################
# PARAMETERS AND SYMBOLS
###############################################################################


threshold_deriv_terms = 3
batch_size_lcm = 2
residue_workers = 32
over_subscribing_ratio = 8
over_subscribing_ratio_final_factor = 1


# Parse command line arguments
parser = argparse.ArgumentParser(description='Compute residues with parallel processing.')
parser.add_argument('-n', type=int, default=9, help='Parameter n (default: 9)')
parser.add_argument('-w', '--max_workers', type=int, default=os.cpu_count(),
                    help='Maximum number of worker processes (default: number of CPUs)')
parser.add_argument('-b', '--batch_size', type=int, default=5,
                    help='Batch size for parallel factorization (default: 5)')
args = parser.parse_args()
n = args.n
max_workers = args.max_workers
batch_size = args.batch_size
print(f"Running with n={n}, max_workers={max_workers}, and batch_size={batch_size}")
## test this script by `python -u residue_compute_separate.py -n6 > log_N6.txt 2>&1` (-u makes sure no buffer when printing)

overall_start = time.time()


# Declare symbols x1 and x2 with the assumptions 0 < x_i < 1.
x1, x2 = symbols('x1 x2', real=True, positive=True)

# Create the list of integration variables: t1, t2, ..., t_{n-1}.
t_vars = symbols(' '.join([f't{i}' for i in range(1, n)]))
if n == 2:
    t_vars = [t_vars]

###############################################################################
# BUILD THE INTEGRAND
###############################################################################

# The overall prefactor:
prefactor = 1/((1 - x1)**n * (1 - x2)**n)

# The integration measure is the product of dt_i/t_i.
measure = 1
for t in t_vars:
    measure *= 1/t

# The product factor:
# For 1 <= k <= r <= n-1, define t_{k,r} = t[k]*...*t[r] and include the factor:
#   (1 - t_{k,r}) / ((1 - x1*t_{k,r})*(1 - x2*t_{k,r})*(1 - x1/t_{k,r})*(1 - x2/t_{k,r}))
prodFactor = 1
for k in range(len(t_vars)):
    for r in range(k, len(t_vars)):
        tkr = 1
        for j in range(k, r+1):
            tkr *= t_vars[j]
        prodFactor *= (1 - tkr) / ((1 - x1*tkr) * (1 - x2*tkr) * (1 - x1/tkr) * (1 - x2/tkr))

# The full integrand is:
integrand = prefactor * measure * prodFactor
f_current = together(integrand)  # simplify the rational function

print("Full integrand (with prefactor) is:")
sympy.pprint(f_current)
print("\n")

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

def candidate_in_unit_circle(candidate, remaining_vars, x_vars):
    """
    Substitute 1 for each remaining integration variable and 0.651 for each x_var,
    then numerically test if Abs(candidate) < 1.
    """
    subs_dict = {var: 1 for var in remaining_vars}
    for xv in x_vars:
        subs_dict[xv] = 0.651
    cand_eval = simplify(candidate.subs(subs_dict))
    try:
        val = float(cand_eval)
        print("Testing candidate pole:", cand_eval, "->", val)
        return val < 1
    except Exception:
        cond = sympy.ask(sympy.Q.lt(cand_eval, 1))
        return True if cond is True else False

def order_by_args(den, v, pole,original_factor):
    test = original_factor.subs(v, pole)
    if test!=0:
        print("Please check the test for variable", v, "at pole", pole, "with the original factor", original_factor)
        print(" test here is", test)
    coeff, fac_list = factor_list(den)
    order = 0
    for fac, exp in fac_list:
        # Divide fac by (v-pole) and check if the factor differs only by a sign.
        ratio = simplify(fac / original_factor)
        if ratio.is_number and (ratio == 1 or ratio == -1):
            order += exp
    return order

def check_term_order(term, v, pole,original_factor):
    _, den = fraction(term)
    test = original_factor.subs(v, pole)
    if test!=0:
        print("Please check the test for variable", v, "at pole", pole, "with the original factor", original_factor)
        print(" test here is", test)
    coeff, fac_list = factor_list(den)
    order = 0
    for fac, exp in fac_list:
        # Divide fac by (v-pole) and check if the factor differs only by a sign.
        ratio = simplify(fac / original_factor)
        if ratio.is_number and (ratio == 1 or ratio == -1):
            order += exp
    return [order,term]


def process_term(term,tmpden,v):
    term_over_den = term / tmpden
    tmp = diff(term_over_den, (v, 1)) # do not do factor here
    return tmp

def my_residue(expr_in, v, pole, original_factor):
    """
    Compute the residue of expr_in in variable v at pole.
    Uses simple inspection of factors for order=1 and the derivative formula when order > 1.
    a revised version, where the m-1 derivatives are taken step by step 
    and the expression factored after each derivatives
    thus we avoid large nested fractions
    Also, in this implementation I avoid doing any together, 
    will leave the sum to the final step
    """
    expr = expr_in
    num, den = fraction(expr)
    # print(f"Starting residue calculation for {v} at pole {pole}")
    # print(f"Original expression numerator: {num}")
    # print(f"Original expression denominator: {den}")
    
    m = order_by_args(den, v, pole, original_factor)
    # print("Computed order for the pole at ", pole, "is", m, " for denominator ", den)
    
    if m == 0:
        # print(f"Order is 0, returning zero")
        return sympy.S.Zero
    elif m == 1:
        t_start = time.time()

        t1 = time.time()
        diff_den = diff(den, v)
        t2 = time.time()
        diff_den_at_pole = diff_den.subs(v, pole)
        t3 = time.time()

        diff_den_factored = factor(diff_den_at_pole)
        t4 = time.time()

        num_at_pole = num.subs(v, pole)
        t5 = time.time()
        if num_at_pole.is_Add and len(num_at_pole.args)>10:
            result = num_at_pole / diff_den_factored
        else:
            result = factor(num_at_pole / diff_den_factored)
        t6 = time.time()

        t_total = t6 - t_start

        # print(f"Order is 1, numerator at pole: {num_at_pole}")
        # print(f"Derivative of denominator at pole: {diff_den}")
        # print(f"Residue result: {result}")
        # print(f"Derivative of denominator at pole: {diff_den_factored}")
        # print(f"Residue result: {result}")

        if t_total > 2.0:
            print("For term", expr, "num_at_pole=",num_at_pole)
            print(f"Timing breakdown for order=1 residue calculation (total: {t_total:.3f}s):")
            print(f"  - Differentiate denominator: {t2-t1:.3f}s")
            print(f"  - Substitute pole value: {t3-t2:.3f}s")
            print(f"  - Factor derivative at pole: {t4-t3:.3f}s")
            print(f"  - Evaluate numerator at pole: {t5-t4:.3f}s")
            print(f"  - Factor final result: {t6-t5:.3f}s")
        return result
    else:
        # print("High order pole at", v, "=", pole, "of order", m)
        # print(f"Using step-by-step derivative approach for order {m}")
        ### the following few lines are the key to make the calculation possible
        ### the built-in residue() function of sympy would take forever to get higher order residues
        ti = time.time()

        # we may still assume the expression passed in is small enough so we can safely do factor in finite time
                
        # Handle large expressions in the first derivative
        tmpnum1, tmpden1 = fraction(factor((v - pole)**m * expr))
        # print(f"First derivative split into fractions")

        # Expand and collect the numerator before checking size
        tmpnum1 = collect(expand(tmpnum1), v)
        # print(f"First derivative numerator expanded and collected")
        
        # If numerator is a large sum, process it in parallel
        if tmpnum1.is_Add and len(tmpnum1.args) > threshold_deriv_terms:
            # print(f"Large first derivative detected with {len(tmpnum1.args)} terms, using parallel processing")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=residue_workers) as executor:
                futures = []
                for term in tmpnum1.args:
                    future = executor.submit(process_term, term, tmpden1, v)
                    futures.append(future)
                
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            t_diff_first = time.time()
            # deriv = sum(results)

            deriv_unfactored = sum(results)
            # print(f"First derivative parallel processing completed")
            t_diff_first = time.time()

            if deriv_unfactored.is_Add:
                # now instead of gether the sum, we factor only each individual terms
                with concurrent.futures.ProcessPoolExecutor(max_workers=residue_workers) as executor:
                    futures = []
                    for term in deriv_unfactored.args:
                        future = executor.submit(factor, term)
                        futures.append(future)
                    
                    myresults = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            deriv = sum(myresults)

            # print(f"First derivative parallel processing completed")
        else:
            # For smaller expressions, use the original approach
            
            # First derivative
            ## doing a factor() before diff() save a lot of time (~2 times speedup)
            deriv_unfactored = diff(factor((v - pole)**m * expr), (v, 1))
            # print(f"First derivative completed")
            t_diff_first = time.time()

            ## doing factor() is still slow (~60 seconds) even if expr numerator is short
            ## there might be a more clever way to do this 
            ##   when there are a lot of terms in deriv_unfactored, this could be slow
            ##   this might be a similar problem to the problem in getting final_results
            # deriv = factor(deriv_unfactored)

            if deriv_unfactored.is_Add:
                # now instead of gether the sum, we factor only each individual terms
                with concurrent.futures.ProcessPoolExecutor(max_workers=residue_workers) as executor:
                    futures = []
                    for term in deriv_unfactored.args:
                        future = executor.submit(factor, term)
                        futures.append(future)
                    
                    myresults = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            deriv = sum(myresults)


            # print(f"First derivative factored: {deriv}")
        
    
        t_factor_first = time.time()
        
        loop_times = []
        loop_diff_times = []
        loop_factor_times = []
        
        # Remaining derivatives
        for ii in range(m-2):
            # print(f"Processing derivative {ii+2} of {m-1}")
            loop_start = time.time()
            
            
            # Process differentiation in parallel if tmpnum is large
            if deriv.is_Add and len(deriv.args) > threshold_deriv_terms:
            # It seems like we never entered this branch for N<=5
                # print(f"Iteration {ii+1}: Large expression detected with {len(deriv.args)} terms, using parallel processing")
                loop_diff_start = time.time()
               
                with concurrent.futures.ProcessPoolExecutor(max_workers=residue_workers) as executor:
                    futures = []
                    for term in deriv.args:
                        future = executor.submit(diff, term, (v,1))
                        futures.append(future)
                    
                    results = [future.result() for future in concurrent.futures.as_completed(futures)]
                
                deriv_unfactored = sum(results)
                # print(f"Iteration {ii+1}: Parallel differentiation completed")
                loop_diff_end = time.time()
                
                    
                if deriv_unfactored.is_Add:
                    # now instead of gether the sum, we factor only each individual terms
                    with concurrent.futures.ProcessPoolExecutor(max_workers=residue_workers) as executor:
                        futures = []
                        for term in deriv_unfactored.args:
                            future = executor.submit(factor, term)
                            futures.append(future)
                        
                        myresults = [future.result() for future in concurrent.futures.as_completed(futures)]
                
                deriv = sum(myresults)

                # print(f"Iteration {ii+1}: Final expression's numerator expanded")
            else:
                # print(f"Iteration {ii+1}: Simple expression with {len(tmpnum.args) if tmpnum.is_Add else 1} terms, differentiation directly")
                deriv_unfactored = diff(deriv, (v, 1))
                loop_diff_end = time.time()
                
                    
                if deriv_unfactored.is_Add:
                    # now instead of gether the sum, we factor only each individual terms
                    with concurrent.futures.ProcessPoolExecutor(max_workers=residue_workers) as executor:
                        futures = []
                        for term in deriv_unfactored.args:
                            future = executor.submit(factor, term)
                            futures.append(future)
                        
                        myresults = [future.result() for future in concurrent.futures.as_completed(futures)]
                
                deriv = sum(myresults)

                # print(f"Iteration {ii+1}: Expression factored")
            
            loop_factor_end = time.time()
            loop_times.append(loop_factor_end - loop_start)
            loop_diff_times.append(loop_diff_end - loop_start)
            loop_factor_times.append(loop_factor_end - loop_diff_end)
        
        t_loop = time.time()
        # print(f"All {m-1} derivatives completed, calculating limit")

        if deriv.is_Add:
            # now instead of gether the sum, we factor only each individual terms
            with concurrent.futures.ProcessPoolExecutor(max_workers=residue_workers) as executor:
                futures = []
                for term in deriv.args:
                    future = executor.submit(limit, term/ factorial(m-1), v, pole)
                    futures.append(future)
                
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            res_unfactored = sum(results)
        else:
            res_unfactored = limit(deriv, v, pole) / factorial(m-1)
        # print(f"Limit calculated: {res_unfactored}")
        t_limit = time.time()
        
        if res_unfactored.is_Add:
            # now instead of gether the sum, we factor only each individual terms
            with concurrent.futures.ProcessPoolExecutor(max_workers=residue_workers) as executor:
                futures = []
                for term in res_unfactored.args:
                    future = executor.submit(factor, term)
                    futures.append(future)
                
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            final_res = sum(results)
        else:
            final_res = factor(res_unfactored)
        # print(f"Final result factored: {final_res}")
        tf = time.time()
        
        if tf-ti > 10:
            print("For term", expr, "\nHigh order pole at", v, "=", pole, "of order", m, " takes total", tf-ti, "seconds")
            print(f"  - diff1: {t_diff_first-ti:.3f}s, factor1: {t_factor_first-t_diff_first:.3f}s, loop: {t_loop-t_factor_first:.3f}s, limit: {t_limit-t_loop:.3f}s, factor2: {tf-t_limit:.3f}s")
            if loop_times:
                loop_str = ", ".join([f"iter{i}: {t:.3f}s (diff: {dt:.3f}s, factor: {ft:.3f}s)" for i, (t, dt, ft) in enumerate(zip(loop_times, loop_diff_times, loop_factor_times))])
                print(f"  - Loop iterations: {loop_str}")
        return final_res
###############################################################################
# ITERATED INTEGRATION: PARALLEL RESIDUE CALCULATION
###############################################################################

integration_order = list(t_vars[::-1])
print("Integration order is:")
print(integration_order)
print("\n")

for i, v in enumerate(integration_order):
    iter_start = time.time()
    print("------------------------------------------------------")
    print(f"Integrating over variable: {v}")
    
    if f_current.is_Add:
        num_terms = len(f_current.args)
    else:
        num_terms = 1
    print("Number of terms in f_current:", num_terms)
    

    # 1. Find common denominator using batched together() approach
    together_start = time.time()

    # This is specifically for finding the common denominator
    def process_denominator_batch(terms_batch):
        """Process a batch of terms to find their common denominator."""
        combined = together(sum(terms_batch))
        _, den = fraction(combined)
        return den

    if f_current.is_Add and len(f_current.args) > batch_size_lcm:
        print(f"Large expression with {len(f_current.args)} terms detected for denominator calculation")
        print(f"Using batch processing with batch_size_lcm={batch_size_lcm}")
        
        # Copy f_current to tmp_current for denominator processing
        tmp_current = f_current
        den_expr = None
        
        # Iteration counter
        iteration = 1
        
        # Process in batches until we get a single denominator
        while tmp_current.is_Add and len(tmp_current.args) > batch_size_lcm:
            batch_start = time.time()
            terms = list(tmp_current.args)
            
            # # Extract denominator from each term
            # terms_denominators = []
            # for arg in tmp_current.args:
            #     _, den = fraction(arg)
            #     terms_denominators.append(1/den)
            # terms = terms_denominators

            num_terms = len(terms)
            num_batches = (num_terms + batch_size_lcm - 1) // batch_size_lcm  # Ceiling division
            
            print(f"Denominator iteration {iteration}: Processing {num_terms} terms in {num_batches} batches")
            
            # Create batches
            batches = []
            for idx_batch in range(0, num_terms, batch_size_lcm):
                batch = terms[idx_batch:idx_batch + batch_size_lcm]
                batches.append(batch)
            
            # Process batches in parallel
            batch_denominators = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for batch in batches:
                    futures.append(executor.submit(process_denominator_batch, batch))
                
                for future in concurrent.futures.as_completed(futures):
                    batch_denominators.append(future.result())
            
            # Create new terms with numerator=1 for each batch denominator
            new_terms = [1/den for den in batch_denominators]
            
            # Update tmp_current with new terms for next iteration
            tmp_current = sum(new_terms)
            
            batch_end = time.time()
            print(f"Denominator iteration {iteration} completed in {batch_end - batch_start:.3f} seconds")
            print(f"Reduced to {len(tmp_current.args) if tmp_current.is_Add else 1} terms")
            
            iteration += 1
        
        # Final denominator calculation
        if tmp_current.is_Add:
            print("Computing final denominator...")
            combined = together(tmp_current)
            _, den_expr = fraction(combined)
        else:
            # Already a single term
            _, den_expr = fraction(tmp_current)
    else:
        # For smaller expressions, use the direct approach
        print("Expression small enough for direct denominator calculation")
        if f_current.is_Add:
            combined = together(f_current)
            _, den_expr = fraction(combined)
        else:
            _, den_expr = fraction(f_current)

    together_end = time.time()
    print(f"Total time for denominator calculation: {together_end - together_start:.3f} seconds")

    # 2. Factor the denominator.
    factor_start = time.time()
    coeff, factors_list = factor_list(den_expr)
    factor_end = time.time()
    print(f"Time to factor denominator: {factor_end - factor_start:.3f} seconds")
    factors = [f for f, exp in factors_list]
    orders = [exp for f, exp in factors_list]
    print("Factors of the denominator:")
    # for fac in factors:
    #     sympy.pprint(fac)
    print(", ".join([str(fac) for fac in factors]))
    print("\n")
    
    # 3. Solve each factor equal to zero for v.
    candidate_poles = []
    original_factors = []  # Store the original factors
    highest_orders = []  # Store highest orders for poles inside the unit circle
    for idx,fac in enumerate(factors):
        sols = solve(fac, v, dict=True)
        for sol in sols:
            if v in sol:
                candidate_poles.append(sol[v])
                original_factors.append(fac)  # Store the corresponding factor
                highest_orders.append(orders[idx])
    # candidate_poles = list(set(candidate_poles))
    # print("Candidate poles for", v, ":")
    # # for cp in candidate_poles:
    # #     sympy.pprint(cp)
    # print(", ".join([str(cp) for cp in candidate_poles]))
    # print("\n")
    
    print("Candidate poles for", v, "with their original factors:")
    for cp, fac, od in zip(candidate_poles, original_factors, highest_orders):
        print(f"Pole: {cp}, Factor: {fac}, Order: {od}")
    print("\n")
    
    # 4. Filter candidates: keep only those inside the unit circle.
    integrated_vars = integration_order[:i+1]  # already integrated variables
    remaining_vars = [var for var in t_vars if var not in integrated_vars]
    x_vars = [x1, x2]
    inside_poles = []
    inside_factors = []  # Store factors for poles inside the unit circle
    inside_highest_orders = []  # Store highest orders for poles inside the unit circle
    print("Testing whether the poles are inside unit circle")
    for idx, cp in enumerate(candidate_poles):
        print("Candidate pole:", cp)
        print("Original factor:", original_factors[idx])
        print("Remaining vars:", remaining_vars)
        if candidate_in_unit_circle(cp, remaining_vars, x_vars):
            inside_poles.append(cp)
            inside_factors.append(original_factors[idx])
            inside_highest_orders.append(highest_orders[idx])
    print("\n")
    print("Poles inside |", v, "| = 1 with their factors and highest orders:")
    for ip, fac, order in zip(inside_poles, inside_factors, highest_orders):
        print(f"Pole: {ip}, Factor: {fac}, Highest Order: {order}")
    print("\n")
    
    # 5. Compute and sum the residues.
    residues = []
    if inside_poles:
        if f_current.is_Add:
            for idx, cp in enumerate(inside_poles):
                if inside_highest_orders[idx] > 1:
                    # Separate terms based on pole order
                    separate_first_order_start = time.time()
                    first_order_terms = []
                    higher_order_terms = []

                    log_memory("at separating poles, before defining executors")
                    
                    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                        # Submit jobs to check pole order for each term
                        # future_to_term = {}
                        term_futures = []
                        for term in f_current.args:
                            term_futures.append( executor.submit(check_term_order, term, v, cp, inside_factors[idx]) )
                            # future_to_term[future] = term
                        
                        log_memory("after submitted checking order jobs")

                        # Process results as they complete
                        for future in concurrent.futures.as_completed(term_futures):
                            order, term = future.result()
                            # term = future_to_term[future]
                            if order == 1:
                                first_order_terms.append(term)
                            else:
                                higher_order_terms.append(term)
                        
                        log_memory("after finished checking order jobs")
                    
                    separate_first_order_end = time.time()
                    print(f"Pole {cp}: Separated {len(first_order_terms)} first-order terms and {len(higher_order_terms)} higher-order terms in {separate_first_order_end - separate_first_order_start:.2f}s")

                    # Process first-order terms (using more workers as these are faster to compute)
                    first_order_start = time.time()
                    first_order_residues = []
                    
                    if first_order_terms:
                        with concurrent.futures.ProcessPoolExecutor(max_workers=int(max_workers)) as executor:
                            # Submit jobs for first-order residues
                            term_futures = []
                            start_time = time.time()
                            
                            for term in first_order_terms:
                                term_futures.append(executor.submit(my_residue, term, v, cp, inside_factors[idx]))
                            
                            # Monitor progress as futures complete
                            completed = 0
                            total = len(term_futures)
                            last_percentage = -1
                            
                            for future in concurrent.futures.as_completed(term_futures):
                                completed += 1
                                current_percentage = int((completed / total) * 100)
                                
                                # Only print when percentage changes by at least 1%
                                if current_percentage > last_percentage:
                                    elapsed_time = time.time() - start_time
                                    
                                    # Estimate remaining time
                                    if completed > 0:
                                        time_per_task = elapsed_time / completed
                                        remaining_tasks = total - completed
                                        remaining_time = time_per_task * remaining_tasks
                                        
                                        print(f"First-order terms progress for pole {cp}: {current_percentage}% ({completed}/{total}) - "
                                              f"Elapsed: {elapsed_time:.1f}s, Est. remaining: {remaining_time:.1f}s")
                                    else:
                                        print(f"First-order terms progress for pole {cp}: {current_percentage}% ({completed}/{total}) - "
                                              f"Elapsed: {elapsed_time:.1f}s")
                                    
                                    last_percentage = current_percentage
                                
                                first_order_residues.append(future.result())
                    
                    first_order_end = time.time()
                    print(f"Processed {len(first_order_terms)} first-order terms in {first_order_end - first_order_start:.2f}s")

                    # Process higher-order terms (using fewer worker groups as these are more compute-intensive)
                    higher_order_start = time.time()
                    higher_order_residues = []
                    
                    if higher_order_terms:
                        with concurrent.futures.ProcessPoolExecutor(max_workers=int(over_subscribing_ratio*max_workers/residue_workers)) as executor:
                            # Submit jobs for higher-order residues
                            term_futures = []
                            start_time = time.time()
                            
                            for term in higher_order_terms:
                                term_futures.append(executor.submit(my_residue, term, v, cp, inside_factors[idx]))
                            
                            # Monitor progress as futures complete
                            completed = 0
                            total = len(term_futures)
                            last_percentage = -1
                            
                            for future in concurrent.futures.as_completed(term_futures):
                                completed += 1
                                current_percentage = int((completed / total) * 100)
                                
                                # Only print when percentage changes by at least 1%
                                if current_percentage > last_percentage:
                                    elapsed_time = time.time() - start_time
                                    
                                    # Estimate remaining time
                                    if completed > 0:
                                        time_per_task = elapsed_time / completed
                                        remaining_tasks = total - completed
                                        remaining_time = time_per_task * remaining_tasks
                                        
                                        print(f"Higher-order terms progress for pole {cp}: {current_percentage}% ({completed}/{total}) - "
                                              f"Elapsed: {elapsed_time:.1f}s, Est. remaining: {remaining_time:.1f}s")
                                    else:
                                        print(f"Higher-order terms progress for pole {cp}: {current_percentage}% ({completed}/{total}) - "
                                              f"Elapsed: {elapsed_time:.1f}s")
                                    
                                    last_percentage = current_percentage
                                
                                higher_order_residues.append(future.result())
                    
                    higher_order_end = time.time()
                    print(f"Processed {len(higher_order_terms)} higher-order terms in {higher_order_end - higher_order_start:.2f}s")
                    
                    # Combine results
                    combine_start = time.time()
                    combined_residues = sum(first_order_residues + higher_order_residues)
                    residues.append(combined_residues)
                    combine_end = time.time()
                    print(f"sum() takes {combine_end - combine_start:.1f}s")
                    print(f"(completed in {combine_end - separate_first_order_start:.1f}s) Combined residue for candidate pole {cp} is: {combined_residues}")
                    
                else:
                    # For poles with only first-order terms, use the original approach
                    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                        term_futures = []  # initialize list for each candidate pole
                        start_time = time.time()  # Record start time
                        
                        for term in f_current.args:
                            term_futures.append(executor.submit(my_residue, term, v, cp, inside_factors[idx]))
                        
                        # Monitor progress as futures complete
                        completed = 0
                        total = len(term_futures)
                        res_values = []
                        last_percentage = -1  # Initialize to ensure first update is printed
                        
                        for future in concurrent.futures.as_completed(term_futures):
                            completed += 1
                            current_percentage = int((completed / total) * 100)
                            
                            # Only print when percentage changes by at least 1%
                            if current_percentage > last_percentage:
                                elapsed_time = time.time() - start_time
                                
                                # Estimate remaining time
                                if completed > 0:
                                    time_per_task = elapsed_time / completed
                                    remaining_tasks = total - completed
                                    remaining_time = time_per_task * remaining_tasks
                                    
                                    print(f"Progress for pole {cp}: {current_percentage}% ({completed}/{total}) - "
                                          f"Elapsed: {elapsed_time:.1f}s, Est. remaining: {remaining_time:.1f}s")
                                else:
                                    print(f"Progress for pole {cp}: {current_percentage}% ({completed}/{total}) - "
                                          f"Elapsed: {elapsed_time:.1f}s")
                                
                                last_percentage = current_percentage
                            
                            res_values.append(future.result())
                        
                        total_time = time.time() - start_time
                        res_sum_cp = sum(res_values)
                        print(f"(completed in {total_time:.1f}s) Residue for candidate pole {cp} is: {res_sum_cp} ")
                        residues.append(res_sum_cp)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_pole = {
                    executor.submit(my_residue, f_current, v, cp, inside_factors[idx]): cp 
                    for idx, cp in enumerate(inside_poles)}
                for future in concurrent.futures.as_completed(future_to_pole):
                    cp = future_to_pole[future]
                    res_val = future.result()
                    print("Residue for candidate pole", cp, "is:", res_val)
                    residues.append(res_val)
    else:
        print(f"No poles found inside the unit circle for variable {v}.")
    
    residue_sum = sum(residues)
    iter_end = time.time()
    print(f"Time taken for integration over variable {v}: {iter_end - iter_start} seconds\n")
    
    # 6. Replace f_current by residue_sum (a function of remaining variables).
    f_current = residue_sum

# # this step is extremely time consuming, costing ~5 hours for N=5
# # there should be a way to parallelize this sum of large number of terms
# final_result = factor(f_current)


# before final factorization, save a intermediate result in mathematica format
# Save intermediate result before final factorization
intermediate_mathematica_filename = f"intermediate_result_N{n}.m"
print(f"Saving intermediate result to Mathematica file: {intermediate_mathematica_filename}")

intermediate_math_start = time.time()

try:
    
    # Open file for writing
    with open(intermediate_mathematica_filename, 'w') as math_file:
        # Write file header
        math_file.write("intermediateResult = ")
        
        # Check if result is a sum or single term
        if f_current.is_Add:
            terms = list(f_current.args)
            num_terms = len(terms)
            print(f"Converting {num_terms} terms to Mathematica format in parallel")
            
            # Write opening bracket for sum
            math_file.write("(")
            
            # Process in batches of max_workers size
            batch_size_intermediate = max_workers
            start_time = time.time()
            completed_terms = 0
            last_percentage = -1
            total_batches = (num_terms + batch_size_intermediate - 1) // batch_size_intermediate
            
            for batch_idx in range(0, num_terms, batch_size_intermediate):
                # Create batch of terms
                end_idx = min(batch_idx + batch_size_intermediate, num_terms)
                terms_batch = terms[batch_idx:end_idx]
                batch_size_actual = len(terms_batch)
                
                # Process batch in parallel
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Convert terms to Mathematica code
                    math_terms = list(executor.map(mathematica_code, terms_batch))
                
                # Write terms to file
                for i, math_term in enumerate(math_terms):
                    if batch_idx > 0 or i > 0:  # Add plus sign except for the first term
                        math_file.write(" + ")
                    math_file.write(math_term)
                    
                    # Flush to disk periodically
                    if (batch_idx + i) % 100 == 0:
                        math_file.flush()
                
                # Update progress
                completed_terms += batch_size_actual
                current_percentage = int((completed_terms / num_terms) * 100)
                
                # Only print when percentage changes by at least 1%
                if current_percentage > last_percentage:
                    elapsed_time = time.time() - start_time
                    batch_num = batch_idx // batch_size_intermediate + 1
                    
                    # Estimate remaining time
                    if completed_terms > 0:
                        time_per_term = elapsed_time / completed_terms
                        remaining_terms = num_terms - completed_terms
                        remaining_time = time_per_term * remaining_terms
                        
                        print(f"Mathematica conversion: {current_percentage}% ({completed_terms}/{num_terms}) - "
                              f"Batch {batch_num}/{total_batches} - "
                              f"Elapsed: {elapsed_time:.1f}s, Est. remaining: {remaining_time:.1f}s")
                    else:
                        print(f"Mathematica conversion: {current_percentage}% ({completed_terms}/{num_terms}) - "
                              f"Elapsed: {elapsed_time:.1f}s")
                    
                    last_percentage = current_percentage
            
            # Write closing bracket for sum
            math_file.write(")")
        else:
            # For a single term, convert directly
            math_file.write(mathematica_code(f_current))
        
        # Write file footer
        math_file.write(";")
    
    intermediate_math_end = time.time()
    print(f"Intermediate result successfully saved to {intermediate_mathematica_filename}")
    print(f"Total intermediate Mathematica export time: {intermediate_math_end - intermediate_math_start:.3f} seconds")
except Exception as e:
    print(f"Error saving intermediate result to Mathematica file: {e}")
    print(f"Exception details: {str(e)}")


def process_batch_old(terms_batch):
    """Process a batch of terms: combine, fraction, expand numerator, and factor."""
    combined = together(sum(terms_batch))
    num, den = fraction(combined)
    expanded_num = expand(num)
    result = factor(expanded_num / den)
    return result

def process_batch(terms_batch):
    if len(terms_batch)>2:
        return optimized_sum(terms_batch)
    elif len(terms_batch)==2:
        return optimized_add(terms_batch[0],terms_batch[1])
    elif len(terms_batch)==1:
        return terms_batch[0]

print("Starting final factorization...")
final_start = time.time()

if f_current.is_Add and len(f_current.args) > batch_size:
    print(f"Large expression detected with {len(f_current.args)} terms")
    print(f"Using batch processing with batch_size={batch_size}")
    
    # Keep processing in batches until the expression is small enough
    iteration = 1
    while f_current.is_Add and len(f_current.args) > batch_size:
        batch_start = time.time()
        terms = list(f_current.args)
        num_terms = len(terms)
        num_batches = (num_terms + batch_size - 1) // batch_size  # Ceiling division
        
        print(f"Iteration {iteration}: Processing {num_terms} terms in {num_batches} batches")
        
        # Create batches
        batches = []
        for i in range(0, num_terms, batch_size):
            batch = terms[i:i + batch_size]
            batches.append(batch)
        
        # Process batches in parallel
        batch_results = []
        start_time = time.time()
        completed_batches = 0
        total_batches = len(batches)
        last_percentage = -1

        with concurrent.futures.ProcessPoolExecutor(max_workers=int(max_workers*over_subscribing_ratio_final_factor)) as executor:
            futures = []
            for batch in batches:
                futures.append(executor.submit(process_batch, batch))
            
            for future in concurrent.futures.as_completed(futures):
                completed_batches += 1
                current_percentage = int((completed_batches / total_batches) * 100)
                
                # Only print when percentage changes by at least 1%
                if current_percentage > last_percentage:
                    elapsed_time = time.time() - start_time
                    
                    # Estimate remaining time
                    if completed_batches > 0:
                        time_per_batch = elapsed_time / completed_batches
                        remaining_batches = total_batches - completed_batches
                        remaining_time = time_per_batch * remaining_batches
                        
                        print(f"Batch progress: {current_percentage}% ({completed_batches}/{total_batches}) - "
                              f"Elapsed: {elapsed_time:.1f}s, Est. remaining: {remaining_time:.1f}s")
                    else:
                        print(f"Batch progress: {current_percentage}% ({completed_batches}/{total_batches}) - "
                              f"Elapsed: {elapsed_time:.1f}s")
                    
                    last_percentage = current_percentage
                
                batch_results.append(future.result())
        
        # Update f_current with the processed results
        batch_results = batch_results[::-1] # avoid a super small expression unprocessed at the end
        f_current = sum(batch_results)
        
        batch_end = time.time()
        print(f"Iteration {iteration} completed in {batch_end - batch_start:.3f} seconds")
        print(f"Reduced to {len(f_current.args) if f_current.is_Add else 1} terms")
        
        iteration += 1
    
    # Final factorization of the reduced expression
    print("Performing final factorization on the reduced expression")
    myterms = list(f_current.args)
    f_current = process_batch(myterms)

    final_result = f_current
    # # factoring a long expression is extremely time consuming
    # final_result = factor(f_current)
else:
    print("Expression is small enough for direct factorization")
    final_result = factor(f_current)

final_end = time.time()
print(f"Final factorization completed in {final_end - final_start:.3f} seconds total")


# Save the result to a Mathematica-compatible file
mathematica_filename = f"result_N{n}.m"
print(f"Saving result to Mathematica file: {mathematica_filename}")

mathematica_start = time.time()

try:
    # Time the conversion to Mathematica code
    conversion_start = time.time()
    mathematica_expr = f"result = {mathematica_code(final_result)};"
    conversion_end = time.time()
    
    # Time the file writing
    writing_start = time.time()
    with open(mathematica_filename, 'w') as f:
        f.write(mathematica_expr)
    writing_end = time.time()
    
    print(f"Mathematica conversion time: {conversion_end - conversion_start:.3f} seconds")
    print(f"File writing time: {writing_end - writing_start:.3f} seconds")
    print(f"Result successfully saved to {mathematica_filename}")
except Exception as e:
    print(f"Error saving to Mathematica file: {e}")

mathematica_end = time.time()
print(f"Total Mathematica export time: {mathematica_end - mathematica_start:.3f} seconds")


overall_end = time.time()
print("======================================================")
print("Final result after all integrations:")
sympy.pprint(final_result)
print(f"Total time taken: {overall_end - overall_start} seconds")
