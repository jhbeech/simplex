import numpy as np
import numpy.typing as npt
from typing import Optional
from typing import Protocol


class Node:
    def __init__(self, vars: list[int], vals: list[int]):
        self.vars = vars
        self.vals = vals

    def return_next_node(self, var, val):
        new_node = self.copy()
        new_node.vars.append(var)
        new_node.vals.append(val)
        return new_node

    def branch(self, new_var):
        return [
            self.return_next_node(new_var, 0),
            self.return_next_node(new_var, 1),
        ]


class LpProblem:
    def __init__(
        self,
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        obj: Optional[float] = None,
        basis_vars: Optional[list[int]] = None,
        basis_vals: Optional[list[float]] = None,
    ) -> None:
        self.A = A
        self.b = b
        self.c = c

    def get_initial_basis(
        self,
    ) -> None:
        A: npt.NDArray[np.float64] = self.A
        self.basis = list(range(A.shape[0], A.shape[1]))

    def primal_simplex(self, max_iterations: int) -> None:
        A = self.A
        b = self.b
        c = self.c
        basis: list[int] = self.basis
        non_basis = get_non_basic(basis, A.shape[0])

        for it in range(max_iterations):
            A_b = A[:, basis]
            A_n = A[:, non_basis]
            c_b = c[basis]
            c_n = c[non_basis]

            reduced_costs = get_reduced_costs(A_b, A_n, c_b, c_n)
            entering_var = get_entering_var_primal(reduced_costs, non_basis)
            leaving_var = get_leaving_var_primal(
                basis, A, A_b, b, entering_var
            )

            print(
                f"iteration: {it}. Objective: {get_objective_value(c_b, A_b, b)}"
            )
            if entering_var is not None:
                basis.remove(leaving_var)
                basis.append(entering_var)
                non_basis.remove(entering_var)
                non_basis.append(leaving_var)
            else:
                print(
                    f"Optimal olution found.{get_objective_value(c_b, A_b, b)}"
                )
                break

        self.obj = get_objective_value(c_b, A_b, b)
        self.basis_vars = basis
        self.basis_vals = get_basis_var_values(A_b, b)

        return self.obj


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
    # prob.gomory_cut_algorithm(gomory_max_cuts)
    upper_bound = prob.obj
    ## round to ints and find obj
    lower_bound = prob.get_rounded_objective()
    queue = [Node(prob.get_most_fractional_var())]
    best_int_node = queue
    while queue:
        branch_node = queue.leftpop()
        for new_node in branch_node.branch():
            queue.remove(new_node)
            ## we are searching this node now
            ## if it is worth searching we will branch into it and add its children otherwise we dont want it

            ## add constraint forcing branch_var = branch_val
            A_branch, b_branch = get_branch_constraints(A, b, branch_node)
            prob = LpProblem(A_branch, b_branch, c)
            new_relaxation = prob.primal_simplex()
            ## probably in fact faster to use current basis and add constraint and do dual simplex??
            new_integral = prob.get_rounded_objective()

            nodes_searched += 1
            if new_relaxation < lower_bound:
                print("prune :)")
                ## by not adding new_node to queue, this branch is gone (i think)
                pass

            else:
                ## keep the node for further searching
                queue.append(new_node)

            if new_integral > lower_bound:
                lower_bound = new_integral
                best_int_node = new_node

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


def get_non_basic(basis: list[int], num_columns: int) -> list[int]:
    return [i for i in range(num_columns) if i not in basis]


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


def get_basis_var_values(
    A_b: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.linalg.inv(A_b) @ b


A = np.array([[1, 1, 1, 0], [2, 1, 0, 1]])
b = np.array([12, 16])
c = np.array([40, 30, 0, 0])

prob = LpProblem(A, b, c)
prob.get_initial_basis()
prob.primal_simplex(10)


"""
problem: A,b,c
basis = initialBasis(A,b)
basis = simplexMethod(A,b,c,basis) -> optimal linear relaxation basis

while continue()
    basis,A,b = generateGomoryCut(basis,A,b,c)
    basis,A,b = dualSimplexMethod(A,b,c,basis) 
"""


## given IP problem A.x <= b, maximize c.x


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
