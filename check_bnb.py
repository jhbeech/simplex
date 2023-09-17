import pulp
import simplex_solver
import time
import numpy as np
import datetime
import matplotlib.pyplot as plt

"""
n,m,max_val,random_seed = 
10,5,10,420 works
(20,20,10,41) works, slow (was singular matrix error needed to switch some 0s for -0.0001)
(20,20,10,40) works now after setting reduced_costs.min() < -0.000001
found issue where was using 999 instead of inf somewhere :/
issue in primal where 
"""
np.random.seed(42)

np.set_printoptions(threshold=np.inf)

## time out for n = m = 30
n = 12
m = 12
max_val = 10


with open(f"./test_logs/{n}_by_{m}_max_val_{max_val}.txt", "a") as f:
    f.write(
        "\n-----------------------------------------------------------------------------------"
    )
    f.write(f"{datetime.datetime.now()}\n")


for it in range(50):
    #     np.random.seed(42)

    # print(k, "----------------------------------")
    A = np.random.randint(0, max_val, size=n * m).reshape(
        (n, m)
    ) / np.random.randint(1, max_val, size=n * m).reshape((n, m))
    x_ = np.random.randint(max_val, size=m)
    delta = np.random.randint(max_val, size=n)
    b = A @ x_ + delta
    c = np.random.randint(max_val, size=m)

    # A = np.array(
    #     [
    #         [1.0, 5.0, 5.0, 9.0, 1.0, 0.0, 0.0, 0.0],
    #         [3.0, 5.0, 1.0, 9.0, 0.0, 1.0, 0.0, 0.0],
    #         [1.0, 9.0, 3.0, 7.0, 0.0, 0.0, 1.0, 0.0],
    #         [6.0, 8.0, 7.0, 4.0, 0.0, 0.0, 0.0, 1.0],
    #     ]
    # )
    # b = np.array([145, 119, 121, 131])
    # c = np.array([6.0, 8.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # print(A.shape)
    # print(b.shape)
    # print(c.shape)

    t0 = time.monotonic()
    prob_pulp = pulp.LpProblem("pulp", pulp.LpMaximize)
    x = pulp.LpVariable.matrix("x", range(A.shape[1]), 0, None, pulp.LpInteger)
    prob_pulp += c @ x

    for lhs, rhs in zip(A @ x, b):
        prob_pulp += lhs <= rhs

    status = prob_pulp.solve()
    t1 = time.monotonic()
    print(pulp.LpStatus[status], "----" * 5)
    print([xi.value() for xi in x])
    obj, best_basis, sols, searched, pruned = simplex_solver.branch_and_bound(
        np.append(A, np.identity(A.shape[0]), axis=1),
        b,
        np.append(c, np.zeros(A.shape[0])),
        1000,
        10000000,
    )
    t2 = time.monotonic()
    print(f"J time: {t2 - t1}. P time: {t1 - t0}")

    print(f"Plp obj: {prob_pulp.objective.value()}")
    print([xi.value() for xi in x])

    plt.plot(sols)
    plt.plot([prob_pulp.objective.value()] * len(sols))
    # plt.show()

    with open(f"./test_logs/{n}_by_{m}_max_val_{max_val}.txt", "a") as f:
        f.write("\n" * 2)
        f.write(f"J time: {t2 - t1}. P time: {t1 - t0}\n")
        print(f"Plp obj: {prob_pulp.objective.value()}\n")
        if prob_pulp.objective.value() != obj:
            f.write(f"{prob_pulp.objective.value()}!={obj}\n")
            f.write("A = np.array(\n")
            for row in A:
                f.write("\t[" + ", ".join(list(map(str, row))) + "],\n")
            f.write(")\n")
            f.write(f"b = np.array([{', '.join(list(map(str,b)))}])\n")
            f.write(f"c = np.array([{', '.join(list(map(str,c)))}])\n")
            # f.write(f"A=np.array({A})\nb=np.array({b})\nc=np.array({c})")
        else:
            f.write(f"{prob_pulp.objective.value()}={obj}\n")
            f.write(f"searched {searched}. pruned {pruned}\n")
            f.write(
                f"number of simplex iterations at optimum {sum(1 if i == obj else 0 for i in sols)} / {len(sols)}\n"
            )
            f.write(f"obj traj: \n\t[{','.join(map(str,sols))}]\n")
