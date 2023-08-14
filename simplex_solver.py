from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Optional
import math
import pyperclip


class Node:
    def __init__(
        self,
        var: int,
        relaxation_val: float,
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        parents_basis: Optional[dict[int, Optional[float]]] = None,
    ):
        self.var = var
        self.relaxation_val = relaxation_val
        self.A = A
        self.b = b
        self.c = c
        self.parents_basis = parents_basis

    def get_branch_problem(self, less_than: bool = True) -> LpProblem:
        """Adds 1 constraint to the linear program based off the branch.
        The constraints associated with the parent's (of the node) values are already contained in A,b.
        if var's value is relaxation_val then adds constraint x_i <= floor(relaxation_val) or - x_i <= -ceil(relaxation_val) depending on value of the arg less_than
        So resulting LpProb is always in form A.x <= b
        """
        A = self.A
        b = self.b
        c = self.c
        num_rows, num_columns = A.shape
        relaxation_val = self.relaxation_val
        var = self.var

        new_column = np.zeros(num_rows)
        new_row = np.zeros(num_columns + 1)

        new_row[var] = 1 if less_than else -1
        new_row[num_columns] = 1  ## new slack variable

        new_b = np.append(
            b,
            (
                math.floor(relaxation_val)
                if less_than
                else -math.ceil(relaxation_val)
            ),
        )

        new_c = np.append(c, 0)

        return LpProblem(
            np.append(
                np.append(A, np.array([new_column]).T, axis=1),
                np.array([new_row]),
                axis=0,
            ),
            new_b,
            new_c,
            basis=self.parents_basis,
        )

    # def return_next_node(self, var: int, val: float):
    #     new_node = self
    #     new_node.vars.append(var)
    #     new_node.vals.append(val)
    #     return new_node


class LpProblem:
    """maximize c^T @ x  subject to A @ x == b
    (A includes slack variables)"""

    def __init__(
        self,
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        basis: Optional[dict[int, Optional[float]]] = None,
        obj: Optional[float] = None,
    ):
        self.A = A
        self.b = b
        self.c = c
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
        # if branch_node:
        # A, b = branch_node.get_branch_constraints()

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
        """selects basic variables are returns the one with the most fractional value (fractional part closest to 0.5)"""
        basis = self.basis
        if not basis:
            return None
        non_slack_basis = sorted(
            [
                (var, val)
                for var, val in basis.items()
                if var < self.A.shape[0]
            ],
            key=lambda x: x[1] % 1,
        )
        return non_slack_basis[0]

    def get_dual_problem(self) -> LpProblem:
        A = self.A
        b = self.b
        c = self.c
        basis_vars = self.get_basis_vars()
        non_slack_vars = [i for i in range(A.shape[1] - A.shape[0])]
        ## transpose A to get dual A. Add in slack variables
        ## slack variable per dual constraint. dual constraint per (non slack) variable in primal problem
        A_dual = np.append(
            A[:, non_slack_vars].T,
            np.identity(len(non_slack_vars)),
            axis=1,
        )
        b_dual = c[non_slack_vars]
        c_dual = -np.append(b, np.zeros(len(non_slack_vars), dtype=np.float64))

        ## exclude last slack variable, it should actully part of the basis, because it has non zero value (assuming the new constraint is violated)

        non_slack_var_vals = c[basis_vars] @ np.linalg.inv(A[:, basis_vars])
        # slack_var_vals = c -
        # return LpProblem(A_dual, b_dual, c_dual, basis)

    def get_basis_vars(self) -> list[int]:
        return list(self.basis.keys())

    def get_slack_vars(self) -> list[int]:
        return get_non_basic(self.get_basis_vars(), self.A.shape[1])

    # def dual_simplex(self, max_iterations: int):
    #     """maintain primal optimality and fix feasibility"""
    #     A = self.A
    #     b = self.b
    #     c = self.c
    #     basis = self.basis
    #     dual_leaving_var = np.argmin(b)

    def primal_simplex(self, max_iterations: int):
        """solves maximize c.x subject to A.x == b, x >=0"""
        A = self.A
        b = self.b
        c = self.c

        basis_vars: list[int] = self.get_basis_vars()
        non_basis_vars = get_non_basic(basis_vars, A.shape[1])
        # print()
        # print(f"{self.basis=}")
        if get_basis_values(A[:, basis_vars], b).min() < 0:
            print("Infeasible.... try dualizing")
            print(
                f"{basis_vars}, vals= {get_basis_values(A[:, basis_vars], b)}"
            )
            self.obj = None
            self.basis = None
            return -999

        for it in range(max_iterations):
            A_b = A[:, basis_vars]
            A_n = A[:, non_basis_vars]
            c_b = c[basis_vars]
            c_n = c[non_basis_vars]
            print(
                f"\riterations: {it}. Objective: {get_objective_value(c_b, A_b, b)}",  # . Basis: {basis_vars}. Vals {get_basis_values(A_b, b)}",  ## remove this for speed
                end="",
            )

            # print(f"{basis_vars=}")
            reduced_costs = get_reduced_costs(A_b, A_n, c_b, c_n)
            entering_var = get_entering_var_primal(
                reduced_costs, non_basis_vars
            )
            leaving_var = get_leaving_var_primal(
                basis_vars, A, A_b, b, entering_var
            )
            # print(f"{basis_vars}")
            # print(f"{get_basis_values(A_b, b)}")
            # print(f"obj={get_objective_value(c_b, A_b, b)}")
            # print(f"{basis_vars=}")
            # print(f"{get_basis_values(A_b,b)}")
            # print(f"{reduced_costs=}")
            # print(f"{entering_var=}")
            # print(f"{leaving_var=}")

            if leaving_var is None and entering_var is not None:
                ## this means that the program is unbounded
                self.obj = None
                self.basis = None
                return -999

            if entering_var is not None:
                basis_vars.remove(leaving_var)
                basis_vars.append(entering_var)
                non_basis_vars.remove(entering_var)
                non_basis_vars.append(leaving_var)

            else:
                print(
                    f"\nOptimal solution found: {get_objective_value(c_b, A_b, b)}"
                )
                break

        self.basis = {
            b: v for b, v in zip(basis_vars, get_basis_values(A_b, b))
        }
        # print(f"{self.basis=}")
        self.obj = get_objective_value(c_b, A_b, b)

        return self.obj

    def dual_simplex(self, max_iterations):
        print("dualize")
        dual_problem = self.get_dual_problem()
        ## move into the basis step into get dual problem
        # dual_problem.basis = {i: None for i in self.get_slack_vars()}
        print("solve")
        dual_problem.primal_simplex(max_iterations)

        print("dualize again (back to original problem")
        self.basis = {i: None for i in dual_problem.get_slack_vars()}
        self.primal_simplex(1)  ## just to set all attrs

    """
    x0 = 1 - y1 - y3
    ##x2 = 10 - 7(x0) - x4
    x2 = 3 + y1 + y3
    
    z = -17.5 (1 - y1 -y3) - 3.5(y1) - 2 (3 + y1 + y3) 
      = - 11.5 - 23y1  - 24y3 
    """

    def get_rounded_objective(self) -> float:
        """if simplex has already been solved, simply rounds all fractional values for x down and finds objective"""
        # for var, val in self.basis.items():
        #     print(f"{var=}", f"c_coff={self.c[var]}", f"val={math.floor(val)}")
        if self.basis:
            return sum(
                self.c[var] * math.floor(val)
                for var, val in self.basis.items()
            )
        else:
            return None


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
    node_var, node_val = prob.get_most_fractional_var()
    queue = [Node(node_var, node_val, A, b, c)]

    best_int_node = queue
    nodes_searched = 0
    while queue:
        print("queue: ", [i.vars for i in queue])
        branch_node = queue.pop(0)  ## this needs to include previous

        # print("---------- new branch var -----------")
        # for branch_node_val in [0, 1, 2, 3]:
        #     branch_node = branch_node_no_val_assigned.copy()
        #     print(
        #         f"nodes searched {nodes_searched}. branching: let x_{branch_node.vars[-1]}={branch_node_val}."
        #     )
        ################# branch node needs to start fres in each iteration here.....................................
        ## branch_node.set_leaf_val(branch_node_val)
        # branch_node.vals[-1] = branch_node_val
        ## we are searching this node now
        ## if it is worth searching we will branch into it and add its children
        ## otherwise we dont want it (its parent is already popped)
        ## add constraint forcing branch_var = branch_val

        # prob = LpProblem(A, b, c)
        # prob.set_initial_basis()
        # prob.print_problem(branch_node)

        # new_relaxation = prob.primal_simplex(
        #     max_iterations_simplex, branch_node
        # )

        ## probably in fact faster to use current basis and add constraint and do dual simplex??
        ## would work if doing depth first tree as we are iteratively adding constraints
        ## in breadth first we need to re-initialise lp problem.
        # new_integral = prob.get_rounded_objective(branch_node)
        # print(f"Integral sol {new_integral}")

        # nodes_searched += 1
        # if new_relaxation < lower_bound:
        #     print("prune :)")
        ## by not adding a child node to queue (for each possible value of branch_node), this branch is gone
        # continue

        # else:
        ## get child node for further searching
        # next_branch_var = prob.get_most_fractional_var()
        # print(f"{next_branch_var=}")
        # if next_branch_var is not None:
        #     queue.append(
        #         branch_node.return_next_node(next_branch_var, -1)
        #     )
        #     print(
        #         f"couldn't prune - searching next most fractional var {queue[-1].vars[-1]}"
        #     )
        # else:
        #     print("nothing else to search thorugh")

        # if new_integral > lower_bound:
        #     lower_bound = new_integral
        #     best_int_node = branch_node_val

        # if upper_bound - lower_bound < gap:
        #     print(
        #         f"bounds close enough, go with best integral sol {lower_bound}"
        #     )
        #     return lower_bound
        # if nodes_searched > max_nodes_to_search:
        #     print(f"exceeded max iterations - best so far: {lower_bound}")
        #     return lower_bound
        # print()

    # print(f"best sol {lower_bound}")

    # return lower_bound


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
    """obj = obj_current_bfs - reduced_costs . x_non_basis
    (x_non_basis = 0)"""
    return c_b @ np.linalg.inv(A_b) @ A_n - c_n


def get_non_basic(basis_vars: list[int], num_columns: int) -> list[int]:
    return [i for i in range(num_columns) if i not in basis_vars]


def get_leaving_var_primal(
    basis: list[int],
    A: npt.NDArray[np.float64],
    A_b: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    entering_var: int,
) -> int:
    if entering_var is None:
        return None
    tableau = np.linalg.inv(A_b) @ A[:, entering_var]

    if tableau.max() < 0:
        print("Uh oh! this program is unbounded")
        return None

    return basis[
        np.argmin(
            np.divide(
                np.linalg.inv(A_b) @ b,
                tableau,
                ## only search over positive elements of tableau
                where=tableau > 0,
                out=np.array([9999999.0] * A_b.shape[0]),
            )
        )
    ]


def get_objective_value(
    c_b: npt.NDArray[np.float64], A_b: npt.NDArray[np.float64], b: np.float64
) -> float:
    return c_b @ np.linalg.inv(A_b) @ b


def get_basis_values(
    A_b: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.linalg.inv(A_b) @ b


def main():
    print("solve basic problem")
    A = np.array([[1, 7, 1, 0], [1, 0, 0, 1]])
    b = np.array([17.5, 3.5])
    c = np.array([1, 10, 0, 0])
    prob = LpProblem(A, b, c)
    prob.set_initial_basis()
    prob.primal_simplex(10000)
    print(f"integral solution {prob.get_rounded_objective()}")

    print("\nadd constraint")
    ## here we are adding a branch constraint, and then using the
    branch_node = Node(1, 2.5, A, b, c, prob.basis)
    branch_problem = branch_node.get_branch_problem(less_than=False)
    branch_problem.print_problem()
    # branch_problem.dual_simplex(1000)

    ## below works. Above is trying to replicate this....
    ## dual simplex method to solve with the new constraint added
    ## from the parent's basis (which is dual feasible but primal infeasible)
    print("-----DUAL---")
    dual_problem = branch_problem.get_dual_problem()
    dual_problem.print_problem()
    dual_problem.primal_simplex(100)

    new_dual_basis_vars = dual_problem.get_basis_vars()
    new_dual_slack_vars = get_non_basic(
        new_dual_basis_vars, dual_problem.A.shape[1]
    )
    new_primal_basis = {i: None for i in new_dual_slack_vars}
    print("--- NEW PRIMAL ---")
    branch_problem.basis = new_primal_basis
    branch_problem.primal_simplex(1000)
    print(f"integral solution {branch_problem.get_rounded_objective()}")

    return None


import time

if __name__ == "__main__":
    tic = time.monotonic()
    main()
    # print("time: ", time.monotonic() - tic)
