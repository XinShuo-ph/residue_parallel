#!/usr/bin/env python3
import time
from sympy import symbols, simplify, together, fraction, diff, Abs, factor_list, solve, factor, multiplicity
from sympy import init_printing
import sympy
import concurrent.futures
import os
import argparse

init_printing()  # for pretty printing if using an interactive environment

###############################################################################
# PARAMETERS AND SYMBOLS
###############################################################################


# Parse command line arguments
parser = argparse.ArgumentParser(description='Compute residues with parallel processing.')
parser.add_argument('-n', type=int, default=9, help='Parameter n (default: 2)')
parser.add_argument('-w', '--max_workers', type=int, default=os.cpu_count(), 
                    help='Maximum number of worker processes (default: number of CPUs)')
args = parser.parse_args()

n = args.n
max_workers = args.max_workers


print(f"Running with n={n} and max_workers={max_workers}")

# Declare symbols x1 and x2 with the assumptions 0 < x_i < 1.
x1, x2 = symbols('x1 x2', real=True, positive=True)
# (We will substitute numerical values later for the free parameters as needed.)

# Create the list of integration variables: t1, t2, ..., t_{n-1}.
t_vars = symbols(' '.join([f't{i}' for i in range(1, n)]))
if n == 2:
    t_vars = [t_vars]
# (For n=3, t_vars is a tuple (t1, t2).)

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
# For 1 <= k <= r <= n-1, define t_{k,r} = t[k]*...*t[r]
# and include the factor:
#      (1 - t_{k,r}) / ((1 - x1*t_{k,r})*(1 - x2*t_{k,r})*(1 - x1/t_{k,r})*(1 - x2/t_{k,r]))
prodFactor = 1
# Note: in Python indexing, t_vars[0] corresponds to t1.
for k in range(len(t_vars)):
    for r in range(k, len(t_vars)):
        # t_{k,r} = product_{j=k}^{r} t_vars[j]
        tkr = 1
        for j in range(k, r+1):
            tkr *= t_vars[j]
        prodFactor *= (1 - tkr) / ((1 - x1*tkr) * (1 - x2*tkr) * (1 - x1/tkr) * (1 - x2/tkr))

# The full integrand is:
integrand = prefactor * measure * prodFactor

# (If desired, you could include an extra overall factor such as 1/(2*pi*i)^(n-1).)
f_current = together(integrand)  # simplify the rational function

print("Full integrand (with prefactor) is:")
sympy.pprint(f_current)
print("\n")

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

def candidate_in_unit_circle(candidate, remaining_vars, x_vars):
    """
    Given a candidate pole (an expression in the current integration variable and possibly other symbols)
    and a list of remaining integration variables (assumed to lie on the unit circle),
    substitute 1 for each remaining variable and a numerical value (here 0.651) for each symbol in x_vars,
    then test if Abs(candidate) < 1.
    Returns True if yes, False otherwise.
    """
    # Build substitution dictionary for remaining integration variables.
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

# --- Custom residue calculation function (for a fraction of polynomials) ---

def order_by_args(den, v, pole):
    """
    Determine the order of the factor (v - pole) in the denominator by examining its factors.
    Factor the denominator using factor_list and then look for a factor that (simplifies to) (v - pole).
    If found, return its exponent; otherwise, return 0.
    """
    coeff, fac_list = factor_list(den)
    order = 0
    for fac, exp in fac_list:
        # Check if fac simplifies to (v - pole)
        if simplify(fac - (v - pole)) == 0:
            order += exp
    return order

def my_residue(expr_in, v, pole):
    """
    Compute the residue of a rational function expr in the variable v at a pole 'pole'.
    
    The function factors the expression and then determines the order m of (v - pole)
    in the denominator by simply inspecting the factors.
    - If m == 0, return 0.
    - If m == 1, return P(pole)/Q'(pole).
    - If m > 1, use the formula:
         1/(m-1)! * lim_{v->pole} d^(m-1)/dv^(m-1)[ (v-pole)^m * expr ].
    """
    expr = factor(expr_in)
    num, den = fraction(expr)
    m = order_by_args(den, v, pole)
    print("Computed order for pole", pole, "=", m)
    if m == 0:
        return sympy.S.Zero
    elif m == 1:
        return factor(num.subs(v, pole) / diff(den, v).subs(v, pole).factor())
    else:
        print("high order poles of ")
        sympy.pprint(expr)
        print("at ",v, " = ", pole, " of order ", m)
        deriv = diff((v - pole)**m * expr, (v, m-1))
        res = limit(deriv, v, pole) / factorial(m-1)
        return factor(simplify(res))


###############################################################################
# TIMING START
###############################################################################
overall_start = time.time()

###############################################################################
# ITERATED INTEGRATION: PARALLEL RESIDUE CALCULATION
###############################################################################

# We integrate in the order: t_{n-1}, t_{n-2}, ..., t_1.
integration_order = list(t_vars[::-1])
print("Integration order is:")
print(integration_order)
print("\n")

# Iterate over the integration variables.
for i, v in enumerate(integration_order):
    iter_start = time.time()
    print("------------------------------------------------------")
    print(f"Integrating over variable: {v}")
    
    # Report the number of terms in f_current.
    if f_current.is_Add:
        num_terms = len(f_current.args)
    else:
        num_terms = 1
    print("Number of terms in f_current:", num_terms)
    
    # 1. Bring f_current into a single rational expression.
    
    num_expr, den_expr = fraction(together(f_current))
    
    # 2. Factor the denominator into irreducible factors.
    coeff, factors_list = factor_list(den_expr)
    factors = [f for f, exp in factors_list for _ in range(exp)]
    print("Factors of the denominator:")
    for fac in factors:
        sympy.pprint(fac)
    print("\n")
    
    # 3. For each factor, solve factor == 0 for v.
    candidate_poles = []
    for fac in factors:
        sols = solve(fac, v, dict=True)
        for sol in sols:
            if v in sol:
                candidate_poles.append(sol[v])
    # Remove duplicates (if any)
    candidate_poles = list(set(candidate_poles))
    print("Candidate poles for", v, ":")
    for cp in candidate_poles:
        sympy.pprint(cp)
    print("\n")
    
    # 4. Filter candidate poles: keep only those with Abs(pole) < 1.
    # Also, require that the remaining integration variables lie on the unit circle.
    integrated_vars = integration_order[:i+1]  # variables already integrated
    remaining_vars = [var for var in t_vars if var not in integrated_vars]
    x_vars = [x1, x2]
    inside_poles = []
    for cp in candidate_poles:
        print("Candidate pole:", cp)
        print("Remaining vars:", remaining_vars)
        if candidate_in_unit_circle(cp, remaining_vars, x_vars):
            inside_poles.append(cp)
    print("Poles inside |", v, "| = 1:")
    for ip in inside_poles:
        sympy.pprint(ip)
    print("\n")
    
    # 5. Compute the sum of the residues at these inside poles.
    residues = []
    if inside_poles:
        # If f_current is a sum (an Add), compute the residue term-by-term.
        if f_current.is_Add:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                for cp in inside_poles:
                    term_futures = []
                    for term in f_current.args:
                        print("Submitting term:", term, "for variable:", v, "with pole:", cp)
                        term_futures.append(executor.submit(my_residue, term, v, cp))
                    # Gather and sum the residues for this candidate pole.
                    res_values = [future.result() for future in term_futures]
                    res_sum_cp = sum(res_values)
                    print("Residue for candidate pole", cp, "is:", res_sum_cp)
                    residues.append(res_sum_cp)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_pole = {executor.submit(my_residue, f_current, v, cp): cp for cp in inside_poles}
                for future in concurrent.futures.as_completed(future_to_pole):
                    cp = future_to_pole[future]
                    res_val = future.result()
                    print("Residue for candidate pole", cp, "is:", res_val)
                    residues.append(res_val)
    else:
        print(f"No poles found inside the unit circle for variable {v}.")
    
    residue_sum = sum(residues)
    # print("Result after integrating over", v, "is:")
    # sympy.pprint(residue_sum)
    iter_end = time.time()
    print(f"Time taken for integration over variable {v}: {iter_end - iter_start} seconds\n")
    
    # 6. Replace f_current by the residue sum (now a function of the remaining variables).
    f_current = residue_sum

final_result = factor(f_current)
overall_end = time.time()
print("======================================================")
print("Final result after all integrations:")
sympy.pprint(final_result)
print(f"Total time taken: {overall_end - overall_start} seconds")
