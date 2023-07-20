import numpy as np
import numpy.typing as npt
from typing import Optional


class Node:
    def __init__(self, vars: list[int], vals: list[float]):
        self.vars = vars
        self.vals = vals

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
        basis: Optional[list[int]] = None,
        basis_vals: Optional[list[float]] = None,
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
                print(f"x_{var} == val")

    def get_initial_basis(self, branch_node: Optional[Node] = None):
        ## maybe should be called set?
        A = self.A
        excluded_vars = branch_node.vars if branch_node else []
        ## optimize?
        self.basis = [
            i for i in range(A.shape[0], A.shape[1]) if i not in excluded_vars
        ]

    def get_most_fractional_var(self) -> Node:
        idx = np.argmin(self.basis_vals)
        return Node([self.basis[idx]], [self.basis_vals[idx]])

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

        basis: list[int] = self.basis
        non_basis = get_non_basic(basis, A.shape[0], branch_node)

        for it in range(max_iterations):
            A_b = A[:, basis]
            A_n = A[:, non_basis]
            c_b = c[basis]
            c_n = c[non_basis]

            print(
                f"iteration: {it}. Objective: {get_objective_value(c_b, A_b, b)}. Basis: {basis}. Vals {get_basis_values(A_b, b)}"  ## remove this for speed
            )

            reduced_costs = get_reduced_costs(A_b, A_n, c_b, c_n)
            entering_var = get_entering_var_primal(reduced_costs, non_basis)
            leaving_var = get_leaving_var_primal(
                basis, A, A_b, b, entering_var
            )

            if entering_var is not None:
                basis.remove(leaving_var)
                basis.append(entering_var)
                non_basis.remove(entering_var)
                non_basis.append(leaving_var)
            else:
                print(
                    f"Optimal solution found: {get_objective_value(c_b, A_b, b)}"
                )
                break

        self.basis = basis
        self.basis_vals = get_basis_values(A_b, b)
        self.obj = get_objective_value(c_b, A_b, b)

        return self.obj

    def get_rounded_objective(self) -> float:
        """if simplex has already been solved, simply rounds all fractional values for x down and finds objective"""
        return np.floor(self.basis_vals) @ self.c[self.basis]


def get_branch_constraints(
    A: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    branch_node: Node,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """returns the A,b such that the values from the node/branch are set."""
    new_b = np.append(b, branch_node.vals)
    A_extra_rows = np.array(
        [
            [1 if j == i else 0 for j in range(A.shape[1])]
            for i in branch_node.vars
        ]
    )

    A_extra_columns = np.zeros(
        (A.shape[0] + len(branch_node.vars), len(branch_node.vars))
    )

    new_A = np.append(
        np.append(A, A_extra_rows, axis=0), A_extra_columns, axis=1
    )
    new_c = np.append(c, np.zeros(len(branch_node.vars)))

    return (new_A, new_b, new_c)


def branch_and_bound(
    A,
    b,
    c,
    gap: float,
    max_nodes_to_search: int,
    gomory_max_cuts: Optional[int] = None,
):
    """branch and bound - assumes all variables are binary lol"""

    prob = LpProblem(A, b, c)
    prob.get_initial_basis()
    prob.primal_simplex(10)
    # prob.gomory_cut_algorithm(gomory_max_cuts)
    upper_bound = prob.obj
    ## round to ints and find obj
    lower_bound = prob.get_rounded_objective()

    ## first in first out
    queue = [prob.get_most_fractional_var()]

    best_int_node = queue
    while queue:
        branch_node = queue.pop(0)
        for branch_node_val in [0, 1]:
            ## branch_node.set_leaf_val(branch_node_val)
            branch_node.vals[-1] = branch_node_val
            ## we are searching this node now
            ## if it is worth searching we will branch into it and add its children
            ## otherwise we dont want it (its parent is already popped)

            ## add constraint forcing branch_var = branch_val
            # A_branch, b_branch, c_branch = get_branch_constraints(
            #     A, b, c, branch_node
            # )
            # prob = LpProblem(A_branch, b_branch, c_branch)
            prob = LpProblem(A, b, c)
            prob.get_initial_basis(branch_node)
            print(f"{prob.basis=}")
            prob.print_problem()

            new_relaxation = prob.primal_simplex(10, branch_node)
            ## probably in fact faster to use current basis and add constraint and do dual simplex??
            ## would work if doing depth first tree as we are iteratively adding constraints
            ## in breadth first we need to re-initialise lp problem.
            new_integral = prob.get_rounded_objective()

            nodes_searched += 1
            if new_relaxation < lower_bound:
                print("prune :)")
                ## by not adding new_node to queue, this branch is gone (i think)
                pass

            else:
                ## keep the node for further searching
                queue.append(branch_node)

            if new_integral > lower_bound:
                lower_bound = new_integral
                best_int_node = branch_node_val

            if (upper_bound - lower_bound < gap) or (
                nodes_searched > max_nodes_to_search
            ):
                print("bounds close enough, go with best integral sol")
                break

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
    basis: list[int], num_columns: int, branch_node: Optional[Node] = None
) -> list[int]:
    ignore_ids = branch_node.vars if branch_node else []
    return [
        i for i in range(num_columns) if i not in basis and i not in ignore_ids
    ]


def get_leaving_var_primal(
    basis: list[int],
    A: npt.NDArray[np.float64],
    A_b: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    entering_var: int,
):
    np.seterr(divide="ignore")
    return (
        basis[
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
    # A = np.array([[1.05, 0.95, 1.5, 0], [2, 1, 0, 1]])
    # b = np.array([12, 16])
    # c = np.array([40, 30, 0, 0])

    A = np.array([[2, 1, 1, 0], [1, 0, 0, 1]])
    b = np.array([10, 15])
    c = np.array([3, 5, 0, 0])
    prob = LpProblem(A, b, c)
    prob.print_problem()
    prob.get_initial_basis()
    prob.primal_simplex(10)
    print("----------------------")

    branch_node = Node([0], [3])
    prob = LpProblem(A, b, c)
    prob.print_problem(branch_node)
    prob.get_initial_basis(branch_node)
    prob.primal_simplex(10, branch_node)
    # print(prob.get_rounded_objective())
    # branch_and_bound(A, b, c, 0.01, 10)

    return None


if __name__ == "__main__":
    main()
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
