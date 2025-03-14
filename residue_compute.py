#!/usr/bin/env python3

import time
from sympy import symbols, expand, simplify, together, fraction, diff, Abs, factor_list, solve, factor, multiplicity, limit, factorial
import sympy
import concurrent.futures
import os
import argparse

sympy.init_printing()  # for pretty printing if using an interactive environment

###############################################################################
# PARAMETERS AND SYMBOLS
###############################################################################

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compute residues with parallel processing.')
parser.add_argument('-n', type=int, default=9, help='Parameter n (default: 9)')
parser.add_argument('-w', '--max_workers', type=int, default=os.cpu_count(),
                    help='Maximum number of worker processes (default: number of CPUs)')
args = parser.parse_args()
n = args.n
max_workers = args.max_workers
print(f"Running with n={n} and max_workers={max_workers}")

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


def my_residue(expr_in, v, pole, original_factor):
    """
    Compute the residue of expr_in in variable v at pole.
    Uses simple inspection of factors for order=1 and the derivative formula when order > 1.
    """
    expr = factor(expr_in)
    num, den = fraction(expr)
    m = order_by_args(den, v, pole, original_factor)
    print("Computed order for the pole at ", pole, "is", m, " for denominator ", den)
    if m == 0:
        return sympy.S.Zero
    elif m == 1:
        return factor(num.subs(v, pole) / diff(den, v).subs(v, pole).factor())
    else:
        # print("High order pole at", v, "=", pole, "of order", m)
        ### the following few lines are the key to make the calculation possible
        ### the built-in residue() function of sympy would take forever to get higher order residues
        ti = time.time()
        deriv = diff((v - pole)**m * expr, (v, m-1))
        t_diff = time.time()
        deriv_factored = factor(deriv)
        t_factor1 = time.time()
        res = limit(deriv_factored, v, pole) / factorial(m-1)
        t_limit = time.time()
        final_res = factor(res)
        tf = time.time()
        print("for term", expr)
        print("High order pole at", v, "=", pole, "of order", m, " takes total", tf-ti, "seconds")
        print(f"  - diff: {t_diff-ti:.3f}s, factor1: {t_factor1-t_diff:.3f}s, limit: {t_limit-t_factor1:.3f}s, factor2: {tf-t_limit:.3f}s")
        return final_res
####  t1 = t_vars[0]
####  t2 = t_vars[1]
####  t3 = t_vars[2]
####  testterm=t1**3*x1**2*x2**4/((-t1 + x1)**5*(-t1 + x2)*(-t1 + x2**2)*(-t1 + x1*x2**2)*(-t1 + x1**2*x2)*(x1 - 1)**5*(x1 + 1)*(x1 - x2)**2*(x2 - 1)**5*(x2 + 1)*(t1*x1 - 1)*(t1*x1 - x2)*(t1*x2 - 1)*(x1*x2 - 1)*(x1*x2**2 - 1)*(x1**2*x2 - 1))


def process_term(term,tmpden,v):
    term_over_den = term / tmpden
    tmp = factor(diff(term_over_den, (v, 1)))
    tmpnum, tmpden = fraction(tmp)
    tmpnum = expand(tmpnum)
    return tmpnum/tmpden

def my_residue_stepbystep(expr_in, v, pole, original_factor):
    """
    Compute the residue of expr_in in variable v at pole.
    Uses simple inspection of factors for order=1 and the derivative formula when order > 1.
    a revised version, where the m-1 derivatives are taken step by step 
    and the expression factored after each derivatives
    thus we avoid large nested fractions
    """
    expr = factor(expr_in)
    num, den = fraction(expr)
    print(f"Starting residue calculation for {v} at pole {pole}")
    print(f"Original expression numerator: {num}")
    print(f"Original expression denominator: {den}")
    
    m = order_by_args(den, v, pole, original_factor)
    print("Computed order for the pole at ", pole, "is", m, " for denominator ", den)
    
    if m == 0:
        print(f"Order is 0, returning zero")
        return sympy.S.Zero
    elif m == 1:
        diff_den = diff(den, v).subs(v, pole).factor()
        num_at_pole = num.subs(v, pole)
        result = factor(num_at_pole / diff_den)
        print(f"Order is 1, numerator at pole: {num_at_pole}")
        print(f"Derivative of denominator at pole: {diff_den}")
        print(f"Residue result: {result}")
        return result
    else:
        print("High order pole at", v, "=", pole, "of order", m)
        print(f"Using step-by-step derivative approach for order {m}")
        ### the following few lines are the key to make the calculation possible
        ### the built-in residue() function of sympy would take forever to get higher order residues
        ti = time.time()
        
        # First derivative
        deriv_unfactored = diff((v - pole)**m * expr, (v, 1))
        print(f"First derivative completed")
        t_diff_first = time.time()
        
        deriv = factor(deriv_unfactored)
        print(f"First derivative factored: {deriv}")
        t_factor_first = time.time()
        
        loop_times = []
        loop_diff_times = []
        loop_factor_times = []
        
        # Remaining derivatives
        for ii in range(m-2):
            print(f"Processing derivative {ii+2} of {m-1}")
            loop_start = time.time()
            
            tmpnum, tmpden = fraction(deriv)
            print(f"Iteration {ii+1}: Expression split into fractions")
            
            tmpnum = collect(expand(tmpnum), v)
            # tmpnum = expand(tmpnum)
            print(f"Iteration {ii+1}: Numerator expanded and collected")
            
            # Process differentiation in parallel if tmpnum is large
            if tmpnum.is_Add and len(tmpnum.args) > 5:
                print(f"Iteration {ii+1}: Large expression detected with {len(tmpnum.args)} terms, using parallel processing")
                loop_diff_start = time.time()
               
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for term in tmpnum.args:
                        future = executor.submit(process_term, term, tmpden, v)
                        futures.append(future)
                    
                    results = [future.result() for future in concurrent.futures.as_completed(futures)]
                
                deriv_unfactored = sum(results)
                print(f"Iteration {ii+1}: Parallel differentiation completed")
                loop_diff_end = time.time()
                
                deriv = together(deriv_unfactored)
                print(f"Iteration {ii+1}: Combined terms by together")
                
                tmpnum, tmpden = fraction(deriv)
                # deriv = collect(tmpnum,v) / tmpden
                deriv = expand(tmpnum) / tmpden
                print(f"Iteration {ii+1}: Final expression's numerator expanded")
            else:
                print(f"Iteration {ii+1}: Simple expression with {len(tmpnum.args) if tmpnum.is_Add else 1} terms, using sequential processing")
                deriv_unfactored = diff(deriv, (v, 1))
                loop_diff_end = time.time()
                
                deriv = factor(deriv_unfactored)
                print(f"Iteration {ii+1}: Expression factored")
            
            loop_factor_end = time.time()
            loop_times.append(loop_factor_end - loop_start)
            loop_diff_times.append(loop_diff_end - loop_start)
            loop_factor_times.append(loop_factor_end - loop_diff_end)
        
        t_loop = time.time()
        print(f"All {m-1} derivatives completed, calculating limit")
        
        res_unfactored = limit(deriv, v, pole) / factorial(m-1)
        print(f"Limit calculated: {res_unfactored}")
        t_limit = time.time()
        
        final_res = factor(res_unfactored)
        print(f"Final result factored: {final_res}")
        tf = time.time()
        
        print("High order pole at", v, "=", pole, "of order", m, " takes total", tf-ti, "seconds")
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
    
    # 1. Write f_current as a single rational expression.
    num_expr, den_expr = fraction(together(f_current))
    
    # 2. Factor the denominator.
    coeff, factors_list = factor_list(den_expr)
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
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                for idx, cp in enumerate(inside_poles):
                    term_futures = []  # initialize list for each candidate pole
                    for term in f_current.args:
                        print("Submitting term:", term, "for variable:", v, "with pole:", cp, "due to factor", inside_factors[idx])
                        term_futures.append(executor.submit(my_residue, term, v, cp, inside_factors[idx]))
                    res_values = [future.result() for future in term_futures]
                    res_sum_cp = sum(res_values)
                    print("Residue for candidate pole", cp, "is:", res_sum_cp)
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

final_result = factor(f_current)
overall_end = time.time()
print("======================================================")
print("Final result after all integrations:")
sympy.pprint(final_result)
print(f"Total time taken: {overall_end - overall_start} seconds")
