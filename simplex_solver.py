import numpy as np
import numpy.typing as npt
from typing import Optional


def simplex(
    A: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
):
    """Maximizes c.x
    Subject to A.x = b.

    A and c include slack variables"""
    basis = get_initial_basis(A, b)
    non_basis = [i for i in range(A.shape[0]) if i not in basis]

    for it in range(1000):
        A_b = A[:, basis]
        A_n = A[:, non_basis]
        c_b = c[basis]
        c_n = c[non_basis]

        reduced_costs = get_reduced_costs(A_b, A_n, c_b, c_n)
        entering_var = get_entering_var(reduced_costs, non_basis)
        leaving_var = get_leaving_var(basis, A_b, b, entering_var)

        print(f"iteration: {it}. Objective: {get_objective_value(c_b, A_b)}")
        if entering_var is not None:
            basis.remove(leaving_var)
            basis.append(entering_var)
            non_basis.remove(entering_var)
            non_basis.append(leaving_var)
        else:
            print("Optimal solution found.")
            break

    return (get_objective_value(c_b, A_b), get_basis_var_values(A_b, b), basis)


def get_initial_basis(
    A: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    return list(range(A.shape[0], A.shape[1]))


def get_entering_var(
    reduced_costs: npt.NDArray[np.float64], non_basis: list[int]
) -> Optional[int]:
    return (
        non_basis[np.argmin(reduced_costs)]
        if reduced_costs.min() < 0
        else None
    )


def get_leaving_var(
    basis: list[int],
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


def get_reduced_costs(
    A_b: npt.NDArray[np.float64],
    A_n: npt.NDArray[np.float64],
    c_b: npt.NDArray[np.float64],
    c_n: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return c_b @ np.linalg.inv(A_b) @ A_n - c_n


def get_objective_value(
    c_b: npt.NDArray[np.float64], A_b: npt.NDArray[np.float64]
) -> float:
    return c_b @ np.linalg.inv(A_b) @ b


def get_basis_var_values(
    A_b: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.linalg.inv(A_b) @ b


A = np.array([[1, 1, 1, 0], [2, 1, 0, 1]])
b = np.array([12, 16])
c = np.array([40, 30, 0, 0])
obj, basis_var_vals, basis = simplex(A, b, c)
print(f"{basis_var_vals=}")
