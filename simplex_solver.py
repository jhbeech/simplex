import numpy as np
import numpy.typing as npt
from typing import Optional
import math


class Node:
    def __init__(self, vars: list[int], vals: list[float]):
        self.vars = vars
        self.vals = vals

    def copy(self):
        return Node(self.vars[:], self.vals[:])

    def return_next_node(self, var: int, val: float):
        new_node = self
        new_node.vars.append(var)
        new_node.vals.append(val)
        return new_node

    ## this should be able to generalise
    # def branch(self, new_var: int):
    #     return [
    #         self.return_next_node(new_var, 0),
    #         self.return_next_node(new_var, 1),
    #     ]


class LpProblem:
    """
    maximize c.T @ x subject to A @ x == b
    (A includes slack variables)"""

    def __init__(
        self,
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        obj: Optional[float] = None,
        basis: Optional[dict[int, Optional[float]]] = None,
        # basis: Optional[list[int]] = None,
        # basis_vals: Optional[list[float]] = None,
    ):
        self.A = A
        self.b = b
        self.c = c

    def print_problem(self, branch_node: Optional[Node] = False):
        c = self.c
        print(
            "Maximize "
            + " + ".join(f"{c[idx]}*x_{idx}" for idx in range(len(c))),
        )
        print("Subject to")
        for constraint_lhs, constraint_rhs in zip(self.A, self.b):
            print(
                " + ".join(
                    f"{constraint_lhs[idx]}*x_{idx}"
                    for idx in range(len(constraint_lhs))
                )
                + f" <= {constraint_rhs}"
            )
        if branch_node:
            print("As well as branch constraints:")
            for var, val in zip(branch_node.vars, branch_node.vals):
                print(f"x_{var} == {val}")

    def set_initial_basis(self):
        """simply sets slack variables to be basic, ie non slack variables are set to 0"""
        A = self.A
        self.basis = {
            i: None for i in range(A.shape[1] - A.shape[0], A.shape[1])
        }

    def get_most_fractional_var(self) -> int:
        """selects basic variables are returns the one with the most fractional value (fractional part closest to 0.5)"""
        basis = self.basis
        if not basis:
            return None
        non_slack_basis_vars = [
            var for var in basis.keys() if var < self.A.shape[0]
        ]
        non_slack_basis_dist_fractional = {
            k: basis[k] % 1 for k in non_slack_basis_vars
        }
        return min(
            non_slack_basis_dist_fractional,
            key=non_slack_basis_dist_fractional.get,
        )

    ## includes vals fixed by currents branch in obj
    def primal_simplex(
        self, max_iterations: int, branch_node: Optional[Node] = None
    ):
        A = self.A
        b = self.b
        c = self.c

        if branch_node:
            vals_fixed_by_branch = np.zeros(A.shape[1])
            vals_fixed_by_branch[branch_node.vars] = branch_node.vals
            b = b - A @ vals_fixed_by_branch
            obj_min = c @ vals_fixed_by_branch
            if b.min() < 0:
                print("infeasible")
                self.obj = None
                self.basis = None
                return -999

            if A.shape[0] == len(branch_node.vars):
                print(
                    f"Branch has fixed all variables, no need for simplex. Obj {obj_min}"
                )
                # print(f"{get_basis_values(A, b)}")

                self.obj = obj_min
                self.basis = None
                return obj_min
        else:
            obj_min = 0

        if b.min() < 0:
            print("infeasible")
            self.obj = None
            self.basis = None
            return -999
        basis_vars: list[int] = list(self.basis.keys())
        non_basis_vars = get_non_basic(basis_vars, A.shape[1], branch_node)

        for it in range(max_iterations):
            A_b = A[:, basis_vars]
            A_n = A[:, non_basis_vars]
            c_b = c[basis_vars]
            c_n = c[non_basis_vars]

            print(
                f"iteration: {it}. Objective: {get_objective_value(c_b, A_b, b)}. Basis: {basis_vars}. Vals {get_basis_values(A_b, b)}"  ## remove this for speed
            )

            # print(f"{basis_vars=}")
            reduced_costs = get_reduced_costs(A_b, A_n, c_b, c_n)
            entering_var = get_entering_var_primal(
                reduced_costs, non_basis_vars
            )
            leaving_var = get_leaving_var_primal(
                basis_vars, A, A_b, b, entering_var
            )

            if entering_var is not None:
                basis_vars.remove(leaving_var)
                basis_vars.append(entering_var)
                non_basis_vars.remove(entering_var)
                non_basis_vars.append(leaving_var)
            else:
                print(
                    f"Optimal solution found: {get_objective_value(c_b, A_b, b) + obj_min}"
                )
                break

        self.basis = {
            b: v for b, v in zip(basis_vars, get_basis_values(A_b, b))
        }
        self.obj = get_objective_value(c_b, A_b, b) + obj_min

        return self.obj

    def get_rounded_objective(
        self, branch_node: Optional[Node] = None
    ) -> float:
        """if simplex has already been solved, simply rounds all fractional values for x down and finds objective"""
        # for var, val in self.basis.items():
        #     print(f"{var=}", f"c_coff={self.c[var]}", f"val={math.floor(val)}")
        if self.basis:
            return sum(
                self.c[var] * math.floor(val)
                for var, val in self.basis.items()
            ) + (
                sum(
                    self.c[var] * val
                    for var, val in zip(branch_node.vars, branch_node.vals)
                )
                if branch_node
                else 0
            )
        else:
            if self.A.shape[0] == len(branch_node.vars):
                vals_fixed_by_branch = np.zeros(self.A.shape[1])
                vals_fixed_by_branch[branch_node.vars] = branch_node.vals
                return self.c @ vals_fixed_by_branch
            else:
                return -999


def branch_and_bound(
    A,
    b,
    c,
    gap: float,
    max_nodes_to_search: int,
    max_iterations_simplex: int,
    # gomory_max_cuts: Optional[int] = None,
):
    """branch and bound - assumes all variables are binary lol"""

    prob = LpProblem(A, b, c)
    prob.set_initial_basis()
    prob.primal_simplex(max_iterations_simplex)
    # prob.gomory_cut_algorithm(gomory_max_cuts)
    upper_bound = prob.obj
    ## round to ints and find obj
    lower_bound = prob.get_rounded_objective()
    print(f"best lower/integral bound {lower_bound}")
    print()
    if upper_bound == lower_bound:
        print(f"First sol is integral {lower_bound}")
        return lower_bound

    ## first in first out
    queue = [Node([prob.get_most_fractional_var()], [-1])]

    best_int_node = queue
    nodes_searched = 0
    while queue:
        print("queue: ", [i.vars for i in queue])
        branch_node_no_val_assigned = queue.pop(0)
        print("---------- new branch var -----------")
        for branch_node_val in [0, 1, 2, 3]:
            branch_node = branch_node_no_val_assigned.copy()
            print(
                f"nodes searched {nodes_searched}. branching: let x_{branch_node.vars[-1]}={branch_node_val}."
            )
            ################# branch node needs to start fres in each iteration here.....................................
            ## branch_node.set_leaf_val(branch_node_val)
            branch_node.vals[-1] = branch_node_val
            ## we are searching this node now
            ## if it is worth searching we will branch into it and add its children
            ## otherwise we dont want it (its parent is already popped)
            ## add constraint forcing branch_var = branch_val

            prob = LpProblem(A, b, c)
            prob.set_initial_basis()
            prob.print_problem(branch_node)

            new_relaxation = prob.primal_simplex(
                max_iterations_simplex, branch_node
            )

            ## probably in fact faster to use current basis and add constraint and do dual simplex??
            ## would work if doing depth first tree as we are iteratively adding constraints
            ## in breadth first we need to re-initialise lp problem.
            new_integral = prob.get_rounded_objective(branch_node)
            print(f"Integral sol {new_integral}")

            nodes_searched += 1
            if new_relaxation < lower_bound:
                print("prune :)")
                ## by not adding a child node to queue (for each possible value of branch_node), this branch is gone
                continue

            else:
                ## get child node for further searching
                next_branch_var = prob.get_most_fractional_var()
                print(f"{next_branch_var=}")
                if next_branch_var is not None:
                    queue.append(
                        branch_node.return_next_node(next_branch_var, -1)
                    )
                    print(
                        f"couldn't prune - searching next most fractional var {queue[-1].vars[-1]}"
                    )
                else:
                    print("nothing else to search thorugh")

            if new_integral > lower_bound:
                lower_bound = new_integral
                best_int_node = branch_node_val

            if upper_bound - lower_bound < gap:
                print(
                    f"bounds close enough, go with best integral sol {lower_bound}"
                )
                return lower_bound
            if nodes_searched > max_nodes_to_search:
                print(f"exceeded max iterations - best so far: {lower_bound}")
                return lower_bound
            print()

    print(f"best sol {lower_bound}")

    return lower_bound


def get_entering_var_primal(
    reduced_costs: npt.NDArray[np.float64],
    non_basis: list[int],
) -> Optional[int]:
    return (
        non_basis[np.argmin(reduced_costs)]
        if reduced_costs.min() < 0
        else None
    )


def get_reduced_costs(
    A_b: npt.NDArray[np.float64],
    A_n: npt.NDArray[np.float64],
    c_b: npt.NDArray[np.float64],
    c_n: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return c_b @ np.linalg.inv(A_b) @ A_n - c_n


def get_non_basic(
    basis_vars: list[int], num_columns: int, branch_node: Optional[Node] = None
) -> list[int]:
    ignore_ids = branch_node.vars if branch_node else []
    return [
        i
        for i in range(num_columns)
        if i not in basis_vars and i not in ignore_ids
    ]


def get_leaving_var_primal(
    basis_vars: list[int],
    A: npt.NDArray[np.float64],
    A_b: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    entering_var: int,
):
    np.seterr(divide="ignore")
    return (
        basis_vars[
            np.argmin(
                np.linalg.inv(A_b)
                @ b
                / ((np.linalg.inv(A_b) @ A[:, entering_var]))
            )
        ]
        if entering_var is not None
        else None
    )


def get_objective_value(
    c_b: npt.NDArray[np.float64], A_b: npt.NDArray[np.float64], b: np.float64
) -> float:
    return c_b @ np.linalg.inv(A_b) @ b


def get_basis_values(
    A_b: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.linalg.inv(A_b) @ b


def main():
    A = np.array([[1, 7, 1, 0], [1, 0, 0, 1]])
    b = np.array([17.5, 3.5])
    c = np.array([1, 10, 0, 0])
    # prob = LpProblem(A, b, c)
    # prob.print_problem()
    # prob.set_initial_basis()
    # prob.primal_simplex(10000)
    # print(f"integral solution {prob.get_rounded_objective()}")

    # branch_node = Node([0], [2])
    # prob.set_initial_basis()
    # prob.print_problem(branch_node)
    # prob.primal_simplex(10, branch_node)
    # print(f"integral solution {prob.get_rounded_objective(branch_node)}")

    # print("-" * 25)
    branch_and_bound(A, b, c, 0.01, 1000, 1000)

    return None


import time

if __name__ == "__main__":
    tic = time.monotonic()
    main()
    print(time.monotonic())

## better to make A smaller rather than bigger
# def get_branch_constraints(
#     A: npt.NDArray[np.float64],
#     b: npt.NDArray[np.float64],
#     c: npt.NDArray[np.float64],
#     branch_node: Node,
# ) -> tuple[
#     npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
# ]:
#     """returns the A,b such that the values from the node/branch are set."""
#     new_b = np.append(b, branch_node.vals)
#     A_extra_rows = np.array(
#         [
#             [1 if j == i else 0 for j in range(A.shape[1])]
#             for i in branch_node.vars
#         ]
#     )

#     A_extra_columns = np.zeros(
#         (A.shape[0] + len(branch_node.vars), len(branch_node.vars))
#     )

#     new_A = np.append(
#         np.append(A, A_extra_rows, axis=0), A_extra_columns, axis=1
#     )
#     new_c = np.append(c, np.zeros(len(branch_node.vars)))

#     return (new_A, new_b, new_c)


"""
problem: A,b,c
basis = initialBasis(A,b)
basis = simplexMethod(A,b,c,basis) -> optimal linear relaxation basis

while continue()
    basis,A,b = generateGomoryCut(basis,A,b,c)
    basis,A,b = dualSimplexMethod(A,b,c,basis) 
"""


"""
class LpProblem
    def gomory_cut_algorithn(self):
        A,b,c = self.A,self.b,self.c
        while continue
            prob = LpProblem(A, b, c)
            prob.get_initial_basis()
            prob.primal_simplex()
            prob.gomory_cut() ## adds constraints and slack variables to A, b
            prob.dual_simplex()
            if prob.basis_values are integral or max_iterations:
                break
        return prob.obj, prob.basis_vars, prob.basis_vals
"""


"""

# max c.x

# let x = [....,1,....]
# A.x = b

# let x = [....,0,....]
# A.x = b
"""
