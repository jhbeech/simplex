from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Optional
import math

MAX_REDUCED_COSTS = 0.0
MIN_REDUCED_COSTS_DUAL_FEASIBILITY = -0.000001
MIN_BASIS_VALS_PRIMAL_FEASIBILITY = -0.00001
TABLEAU_MIN_PIVOT_COL_DUAL_SIMPLEX = -0.00001
TABLEAU_MIN_PIVOT_COL_PRIMAL_SIMPLEX = 0.00001


class LpProblem:
    """maximize c^T @ x subjects to x>=0 & (A @ x == b if slacks_included else A @ x <= b)"""

    def __init__(
        self,
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        basis: Optional[dict[int, Optional[float]]] = None,
        obj: Optional[float] = None,
        slacks_included: bool = True,
    ):
        if not slacks_included:
            n_non_slack_vars = A.shape[1]
            n_slack_vars = A.shape[0]
            A = np.append(A, np.identity(n_slack_vars), axis=1)
            c = np.append(c, np.zeros(n_slack_vars))
        else:
            n_slack_vars = A.shape[0]
            n_non_slack_vars = A.shape[1] - n_slack_vars
        self.A = A
        self.b = b
        self.c = c
        self.n_slack_vars = n_slack_vars
        self.n_non_slack_vars = n_non_slack_vars
        self.obj = obj
        self.basis = basis

    def print_problem(self, dual=False):
        letter = "y" if dual else "x"
        c = self.c
        print(
            "Maximize "
            + " + ".join(f"{c[idx]}*{letter}_{idx}" for idx in range(len(c))),
        )
        print("Subject to")
        for constraint_lhs, constraint_rhs in zip(self.A, self.b):
            print(
                " + ".join(
                    f"{constraint_lhs[idx]}*{letter}_{idx}"
                    for idx in range(len(constraint_lhs))
                )
                + f" == {constraint_rhs}"
            )

    def set_initial_basis(self):
        """simply sets slack variables to be basic, ie non slack variables are set to 0"""
        A = self.A
        self.basis = {
            i: None for i in range(A.shape[1] - A.shape[0], A.shape[1])
        }

    def get_most_fractional_var(self) -> tuple[int, float]:
        """selects basic variable with the most fractional value (fractional part closest to 0.5)"""
        basis = self.basis
        if basis is None:
            return None
        non_slack_basis = sorted(
            [
                (var, val)
                for var, val in basis.items()
                if var < self.n_non_slack_vars
            ],
            key=lambda x: abs(x[1] % 1.0 - 0.5),
        )
        var, val = non_slack_basis[0]

        return (
            (var, val)
            if not math.isclose(val, round(val), abs_tol=1e-5)
            else (None, None)
        )

    def get_basis_vars(self) -> list[int]:
        return list(self.basis.keys())

    def primal_simplex(
        self,
        max_iterations: int,
        blands_rule: bool = False,
        log_iterations: bool = True,
    ) -> float:
        """maximizes lp using revised simplex algorithm"""
        A = self.A
        b = self.b
        c = self.c
        basis_vars: list[int] = self.get_basis_vars()
        non_basis_vars = get_non_basic(basis_vars, A.shape[1])
        if get_basis_values(A[:, basis_vars], b).min() < 0:
            print("Infeasible.... try dualizing")
            print(
                f"{basis_vars}, vals= {get_basis_values(A[:, basis_vars], b)}\n"
            )
            self.obj = None
            self.basis = None
            return -999.0

        A_b_inv = None
        entering_var = None
        leaving_var = None
        leaving_var_loc = None
        for it in range(max_iterations):
            A_b = A[:, basis_vars]
            A_b_inv = get_inv_sherman_morrison(
                A, A_b, A_b_inv, entering_var, leaving_var, leaving_var_loc, it
            )
            A_n = A[:, non_basis_vars]
            c_b = c[basis_vars]
            c_n = c[non_basis_vars]
            if log_iterations:
                print(
                    f"\riterations: {it}. Objective: {get_objective_value(c_b, A_b_inv, b)}",  # , . Basis: {basis_vars}.",  # Vals {get_basis_values(A_b, b)}",
                    end="",
                )
            reduced_costs = get_reduced_costs(A_b_inv, A_n, c_b, c_n)
            entering_var = get_entering_var_primal(
                reduced_costs, non_basis_vars, blands_rule
            )
            leaving_var = get_leaving_var_primal(
                basis_vars, A, A_b_inv, b, entering_var
            )

            if leaving_var is None and entering_var is not None:
                ## this means that the program is unbounded
                self.obj = None
                self.basis = None
                return -999.0

            if entering_var is not None and it < max_iterations - 1:
                leaving_var_loc = basis_vars.index(leaving_var)
                basis_vars[leaving_var_loc] = entering_var
                ## need to replace leaving var with entering var (ie they are in same loc)
                ## for sherman morrison algorithm to work
                entering_var_loc = non_basis_vars.index(entering_var)
                non_basis_vars[entering_var_loc] = leaving_var

            else:
                if log_iterations:
                    print(
                        f"\nOptimal solution found: {get_objective_value(c_b, A_b_inv, b)}"
                    )
                break
        self.basis = {
            b: v for b, v in zip(basis_vars, get_basis_values(A_b_inv, b))
        }
        self.obj = get_objective_value(c_b, A_b_inv, b)

        return self.obj

    def dual_simplex(
        self, max_iterations: int, blands_rule: bool = False
    ) -> float:
        """uses the dual simplex method to solve the LP with an added constraint making it primal infeasible (but dual feasible).
        This is equivalent to solving min b.y subject to A.T @ y >= c"""
        A = self.A
        b = self.b
        c = self.c
        basis_vars: list[int] = self.get_basis_vars()
        non_basis_vars = get_non_basic(basis_vars, A.shape[1])
        reduced_costs = get_reduced_costs(
            A[:, basis_vars],
            A[:, non_basis_vars],
            c[basis_vars],
            c[non_basis_vars],
        )

        if reduced_costs.min() < MIN_REDUCED_COSTS_DUAL_FEASIBILITY:
            print("Dual Infeasible.... try primal simplex")
            self.obj = None
            self.basis = None
            print(f"{reduced_costs=}")
            return -999.0

        A_b_inv = None
        entering_var = None
        leaving_var = None
        leaving_var_loc = None
        for it in range(max_iterations):
            A_b = A[:, basis_vars]
            A_b_inv = get_inv_sherman_morrison(
                A, A_b, A_b_inv, entering_var, leaving_var, leaving_var_loc, it
            )
            A_n = A[:, non_basis_vars]
            c_b = c[basis_vars]
            c_n = c[non_basis_vars]

            print(
                f"\r{it}/{max_iterations} obj: {get_objective_value(c_b,A_b_inv,b)}",
                end="",
            )
            reduced_costs = get_reduced_costs(A_b_inv, A_n, c_b, c_n)
            basis_vals = A_b_inv @ b
            ## TODO
            # if blands_rule:
            ## use the left most index with negative basis val
            # leaving_var_loc = np.argwhere(basis_vals < 0.0001).min()
            # else:
            leaving_var_loc = np.argmin(basis_vals)
            tableau = A_b_inv[leaving_var_loc, :] @ A_n
            if basis_vals.min() > MIN_BASIS_VALS_PRIMAL_FEASIBILITY:
                ## problem now feasible
                break

            if tableau.min() >= TABLEAU_MIN_PIVOT_COL_DUAL_SIMPLEX:
                print(
                    "Uh oh! this problem is dual unbounded (primal infeasible)"
                )
                self.basis = None
                self.obj = None
                return -999.0
            leaving_var = basis_vars[leaving_var_loc]
            # entering_var = non_basis_vars[leaving_var]
            entering_var_loc = np.argmin(
                np.divide(
                    reduced_costs,
                    -tableau,
                    out=np.array([np.inf] * len(reduced_costs)),
                    where=tableau < TABLEAU_MIN_PIVOT_COL_DUAL_SIMPLEX,
                )
            )
            entering_var = non_basis_vars[entering_var_loc]
            basis_vars[leaving_var_loc] = entering_var
            ## need to replace leaving var with entering var (ie they are in same loc)
            ## for sherman morrison algorithm to work
            non_basis_vars[entering_var_loc] = leaving_var
        print()
        self.basis = {
            b: v for b, v in zip(basis_vars, get_basis_values(A_b_inv, b))
        }
        self.obj = get_objective_value(c_b, A_b_inv, b)
        return self.obj

    def get_rounded_objective(self) -> Optional[float]:
        """Rounds all fractional values for x down and finds objective"""
        if self.basis:
            return sum(
                self.c[var] * math.floor(round(val, 6))  ## TODO
                for var, val in self.basis.items()
            )
        else:
            return None

    def get_branch_problems(self) -> Optional[list[LpProblem]]:
        """branches on most fractional var xf = vf and
        returns two LpProblems - original LP + xf <= floor(vf) or
        xf >= ceil(vf)"""
        branch_probs: list[LpProblem] = []
        A = self.A.copy()
        b = self.b.copy()
        c = self.c.copy()
        num_rows, num_columns = A.shape
        var, relaxation_val = self.get_most_fractional_var()
        if var is None:
            return None
        for less_than in [True, False]:
            new_column = np.zeros(num_rows)
            new_row = np.zeros(num_columns + 1)

            new_row[var] = 1 if less_than else -1
            new_row[num_columns] = 1  ## new slack variable

            new_b = np.append(
                b,
                (
                    math.floor(round(relaxation_val, 6))
                    if less_than
                    else -math.ceil(round(relaxation_val, 6))
                ),
            )

            new_c = np.append(c, 0)
            new_basis = self.basis.copy()
            new_basis.update(
                {
                    num_columns: -relaxation_val
                    + math.floor(round(relaxation_val, 6))
                    if less_than
                    else relaxation_val - math.ceil(round(relaxation_val, 6))
                }
            )
            branch_probs.append(
                LpProblem(
                    np.append(
                        np.append(A, np.array([new_column]).T, axis=1),
                        np.array([new_row]),
                        axis=0,
                    ),
                    new_b,
                    new_c,
                    basis=new_basis,
                )
            )
        return branch_probs

    def add_gomory_cut_constraint(self) -> None:
        """adds gomory cut constraint to LpProblem inplace"""
        A = self.A
        b = self.b
        c = self.c
        frac_var, _ = self.get_most_fractional_var()
        if frac_var is None:
            return 0
        basis_vars = self.get_basis_vars()
        A_b = A[:, basis_vars]
        ## maybe let's save A_b to self
        ## at the end of simplex instead
        A_b_inv = np.linalg.inv(A_b)
        b_i = A_b_inv[frac_var] @ b
        num_vars = A.shape[1]
        non_basis_vars = get_non_basic(basis_vars, num_vars)
        a_ = A_b_inv[frac_var, :] @ A[:, non_basis_vars]

        A_new_row = np.zeros(num_vars)
        A_new_row[non_basis_vars] = np.floor(a_) - a_
        b_new_val = np.floor(b_i) - b_i

        A_new_col = np.zeros(A.shape[0] + 1)
        A_new_col[A.shape[0]] = 1
        A = np.append(
            np.append(A, np.array([A_new_row]), axis=0),
            np.array([A_new_col]).T,
            axis=1,
        )
        b = np.append(b, b_new_val)
        c = np.append(c, 0)

        self.basis[A.shape[0]] = None
        self.A = A
        self.b = b
        self.c = c

        return 1


def branch_and_bound(
    A: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    # max_nodes_to_search: int,
    max_iterations_simplex: int,
    max_iterations_dual_simplex: int,
    # slacks_included: bool = True,
    # gomory_max_cuts: Optional[int] = None,
    mip_gap_threshold: float = 0,
) -> tuple[float, npt.NDArray[np.float64], list[tuple[int, float]], int, int]:
    """branch and bound - assumes all variables are integers"""
    ## TODO - have upper bound decrease if possible.
    ## could terminate if gap is small enough
    ## TODO - add gomory cuts
    integral_sols = []
    relaxation_sols = []
    prob = LpProblem(A, b, c)
    prob.set_initial_basis()
    prob.primal_simplex(max_iterations_simplex)
    upper_bound = prob.obj
    ## round (floor) to ints and find obj
    lower_bound = prob.get_rounded_objective()
    print(f"best lower/integral bound {lower_bound}")
    if upper_bound == lower_bound:
        sol = np.floor(
            np.array(
                [
                    round(best_basis.get(var, 0), 6)
                    for var in range(prob.n_non_slack_vars)
                ]
            )
        )
        print(f"First sol is integral {lower_bound}")
        return lower_bound, sol, [(lower_bound, lower_bound)], 0, 0

    queue = prob.get_branch_problems()
    if queue is None:
        print("no fractional values")
        return lower_bound, upper_bound, 0, 0
    best_basis = prob.basis
    nodes_searched = 0
    branches_pruned = 0
    while queue:
        nodes_searched += 1
        branch_prob = queue.pop(0)
        branch_prob.dual_simplex(max_iterations_dual_simplex)
        relaxation = branch_prob.obj
        integral = branch_prob.get_rounded_objective()
        print(f"relaxation {relaxation}. integral {integral}")
        print(f"lower bound {lower_bound} upper bound {upper_bound}")

        if relaxation is None or relaxation < lower_bound:
            print("pruning :D")
            branches_pruned += 1
            continue
        else:
            print("cant prune as relaxation better than lower bound")
            child_probs = branch_prob.get_branch_problems()
            if child_probs is None:
                print("all vars are ints :), nothing to add to queue")
            elif integral > relaxation - mip_gap_threshold:
                print("within mip gap threshold, nothing to add to queue")
            else:
                queue += child_probs
        if integral > lower_bound:
            ## lower bound is best integral solution
            lower_bound = integral
            best_basis = branch_prob.basis
        integral_sols.append(lower_bound)
        relaxation_sols.append(relaxation)
    print(f"{nodes_searched=}")
    print(f"{branches_pruned=}")
    print("final lower bound, upper bound", lower_bound, upper_bound)
    sol = np.floor(
        np.array(
            [
                round(best_basis.get(var, 0), 6)
                for var in range(prob.n_non_slack_vars)
            ]
        )
    )
    return (
        lower_bound,
        sol,
        list(zip(relaxation_sols, integral_sols)),
        nodes_searched,
        branches_pruned,
    )


def get_entering_var_primal(
    reduced_costs: npt.NDArray[np.float64],
    non_basis: list[int],
    blands_rule: bool,
) -> Optional[int]:
    if blands_rule:
        np.argwhere(reduced_costs < MAX_REDUCED_COSTS)
        return (
            min(
                non_basis[i]
                for i in np.argwhere(reduced_costs < MAX_REDUCED_COSTS).T[0]
            )
            if reduced_costs.min() < MAX_REDUCED_COSTS
            else None
        )
    else:
        return (
            non_basis[np.argmin(reduced_costs)]
            if reduced_costs.min() < MAX_REDUCED_COSTS
            else None
        )


def get_reduced_costs(
    A_b_inv: npt.NDArray[np.float64],
    A_n: npt.NDArray[np.float64],
    c_b: npt.NDArray[np.float64],
    c_n: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """obj = obj_current_bfs - reduced_costs . x_non_basis
    (x_non_basis = 0). So negative reduced costs -> bring into basis"""
    return c_b @ A_b_inv @ A_n - c_n


def get_non_basic(basis_vars: list[int], num_columns: int) -> list[int]:
    return [i for i in range(num_columns) if i not in basis_vars]


def get_inv_sherman_morrison(
    A: npt.NDArray[np.float64],
    A_b: npt.NDArray[np.float64],
    B_inv: npt.NDArray[np.float64],
    entering_var_idx: npt.NDArray[np.float64],
    leaving_var_idx: int,
    leaving_var_loc: int,
    iteration: int,
) -> npt.NDArray[np.float64]:
    """use sherman morrison formula to get inverse of A_b"""
    if B_inv is None or iteration % 50 == 0:
        return np.linalg.inv(A_b)
    u = A[:, entering_var_idx] - A[:, leaving_var_idx]
    v = np.zeros(A.shape[0])
    v[leaving_var_loc] = 1
    return B_inv - (B_inv @ np.array([u]).T) @ (np.array([v]) @ B_inv) / (
        1 + v.T @ B_inv @ u
    )


def get_leaving_var_primal(
    basis: list[int],
    A: npt.NDArray[np.float64],
    A_b_inv: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    entering_var: int,
) -> int:
    if entering_var is None:
        return None
    tableau = A_b_inv @ A[:, entering_var]
    # print(f"{tableau=}")
    if tableau.max() < 0:
        print("\nUh oh! this program is unbounded")
        return None

    return basis[
        np.argmin(
            np.divide(
                A_b_inv @ b,
                tableau,
                ## only search over positive elements of tableau
                where=tableau >= TABLEAU_MIN_PIVOT_COL_PRIMAL_SIMPLEX,
                out=np.array([np.inf] * A_b_inv.shape[0]),
            )
        )
    ]


def get_objective_value(
    c_b: npt.NDArray[np.float64],
    A_b_inv: npt.NDArray[np.float64],
    b: np.float64,
) -> float:
    return c_b @ A_b_inv @ b


def get_basis_values(
    A_b_inv: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return A_b_inv @ b


def check_gomory():
    max_val = 10
    n = 5
    m = 5
    np.random.seed(42)
    A = np.random.randint(max_val, size=n * m).reshape((n, m))
    x_ = np.random.randint(max_val, size=m)
    delta = np.random.randint(max_val, size=n)
    b = A @ x_ + delta
    c = np.random.randint(max_val, size=m)
    A = np.append(A, np.identity(A.shape[0]), axis=1)
    c = np.append(c, np.zeros(A.shape[0]))
    prob = LpProblem(A, b, c, slacks_included=True)
    prob.set_initial_basis()
    prob.primal_simplex(100)
    print(prob.basis)

    print(prob.get_most_fractional_var())
    # prob.obj
    for _ in range(100):
        prob.add_gomory_cut_constraint()
        prob.primal_simplex(100)
        print(prob.basis)


def check_bnb(n, m, max_val):
    import pulp
    import time
    import matplotlib.pyplot as plt

    A = np.random.randint(max_val, size=n * m).reshape((n, m))
    x_ = np.random.randint(max_val, size=m)
    delta = np.random.randint(max_val, size=n)
    b = A @ x_ + delta
    c = np.random.randint(max_val, size=m)
    A = np.append(A, np.identity(A.shape[0]), axis=1)
    c = np.append(c, np.zeros(A.shape[0]))
    t0 = time.monotonic()
    prob_pulp = pulp.LpProblem("pulp", pulp.LpMaximize)
    x = pulp.LpVariable.matrix("x", range(A.shape[1]), 0, None, pulp.LpInteger)
    prob_pulp += c @ x
    for lhs, rhs in zip(A @ x, b):
        prob_pulp += lhs == rhs
    status = prob_pulp.solve(pulp.PULP_CBC_CMD())
    t1 = time.monotonic()
    (
        obj,
        var_values,
        trajectory,
        nodes_searched,
        branches_pruned,
    ) = branch_and_bound(A, b, c, 1000000, 100000)
    t2 = time.monotonic()
    print("\n\n")
    print(f"obj: {obj}. time: {t2 - t1}. ")
    print(f"pulp obj: {prob_pulp.objective.value()} pulp time: {t1 - t0}")
    print(
        f"searched {nodes_searched} nodes. pruned {branches_pruned} branches."
    )
    plt.plot(trajectory)
    plt.plot([prob_pulp.objective.value()] * len(trajectory))
    plt.show()
    return None


if __name__ == "__main__":
    np.random.seed(42)
    check_bnb(25, 25, 100)
    # np.random.seed(42)
    # n = 5
    # m = 5
    # max_val = 10
    # A = np.random.randint(max_val, size=n * m).reshape((n, m))
    # x_ = np.random.randint(max_val, size=m)
    # delta = np.random.randint(max_val, size=n)
    # b = A @ x_ + delta
    # c = np.random.randint(max_val, size=m)
    # A = np.append(A, np.identity(A.shape[0]), axis=1)
    # c = np.append(c, np.zeros(A.shape[0]))

    # basis_vars = list(range(n))
    # B = A[:, basis_vars]
    # B_inv = np.linalg.inv(A[:, basis_vars])

    # entering_var_idx = 6
    # leaving_var_idx = 2
    # leaving_var_loc = 2

    # u = A[:, entering_var_idx] - A[:, leaving_var_idx]
    # v = np.array([1 if i == leaving_var_loc else 0 for i in range(m)])

    # B_new = B + np.outer(u, v)

    # print(np.linalg.inv(B_new))
    # print()
    # print(
    #     B_inv
    #     - (B_inv @ np.array([u]).T)
    #     @ (np.array([v]) @ B_inv)
    #     / (1 + v.T @ B_inv @ u)
    # )
    # print()

    # print(
    #     get_inv_sherman_morrison(
    #         A, B_inv, entering_var_idx, leaving_var_idx, leaving_var_loc
    #     )
    # )
