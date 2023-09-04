# The Simplex Algorithm

Maximize
$c^T x$

Subject to $Ax = b, x \geq 0 $.

For some basis, B, we have:

$$
\max{c_B^T x_B + c_N^T x_N}, \\
A_B x_B + A_N x_N = b, x \geq 0 
$$

Rearranging:

$$
\begin{aligned}
x_B &= A_B^{-1} (b - A_N x_N) = A_B^{-1}b \\
\text{reduced costs} &= c_B^T A_B^{-1}A_N − c_N^T \\
z &= c_B^T A_B^{-1}b - \text{reduced costs}   \cdot c_N,\\ c_N &= 0
\end{aligned}
$$


## Reduced costs $\rightarrow$ Entering variable
If the reduced cost of a non basic variable $x_E \in x_N$ is negative, then we can increase z by giving $c_N$ a positive value, so it 'enters the basis'. 

```
reduced_costs = c[basis] @ A[,:basis].inv() A[:,nonbasis] - c[nonbasis]
## getting actual idx, not iloc
E = idxmin(
    reduced_costs,
    return_null_if=reduced_costs.isnull()
)
```

## Leaving Variable and Ratio Test
We need to select the 'leaving variable' $x_L$ so that we maintain feasibility (keep variables greater than 0). 

Consider only the rows of the tableau where the coefficient of the entering var is negative (if not we have an infeasible program)

$$
\begin{aligned}
x_1 &= \alpha_1 - \beta_1 x_E + ... \\
x_2 &= \alpha_2 - \beta_2 x_E + ... \\
\alpha_1 / \beta_1  &\lt \alpha_2 / \beta_2, (\alpha_i \gt 0, \beta_i > 0) \\
\end{aligned}
$$

Then $\Rightarrow$

$$
\begin{aligned}
x_E &= \alpha_1 / \beta_1 -x_1 / \beta_1 + ...\\
x_2 &= \alpha_2 - \beta_2 / \beta_1 \alpha_1 - \beta_2 x_1/\beta_1 + ...
\end{aligned}
$$

This solution is feasible if the constants are positive. The constant term for the $x_E$ expression is clearly positive. Taking $x_2$'s constant term and dividing by $\beta_2$, we have: 

$$
( \alpha_2 - \beta_2 / \beta_1 \alpha_1 ) / \beta_2 = \alpha_2/\beta_2 - \alpha_1 / \beta_1 \gt 0
$$

Hence the new solution is feasible. 


## Implementing this Method
The above gives us a basic outline of the simplex method: 
- We take a feasible basis
- calculate the reduced costs to get the entering variable
- Use the ratio test described above to calculate the leaving variable.

### Initial Feasible Solution
If our LP is of the form 
Maximize
$c \cdot x$
subject to $A x \leq b$, $b\geq\vec{0}$, then we always have an easy initial feasible solution at our disposal: $x=\vec{0}$

### Entering Variable - Reduced Costs
Since our expression for $z$ after choosing our initial basis is:

$$
z = c_B^T A_B^{-1}b -(c_B^T A_B^{-1}A_N − c_N^T )  \cdot c_N
$$

We can read of our reduced costs (in the brackets). If all the reduced costs are positive, then our solution is optimal. If we have a negative reduced cost then we can increase the objective by including the corresponding variable in the basis.




### Leaving Variable - Ratio Test

$$
x_B = A_B^{-1} b - A_B^{-1}A_N x_N\\
$$

Removing matrix notation [ie let $(A_B^{-1})_i = \alpha_i$] and focusing on the entering variable,

$$
x_i = \alpha_i - \beta_i x_E - ....... \text{, for }i \in B
$$


As seen in the above analysis, we look at all the constraints with negative coefficients of $x_E$, compare with the constant values. Taking the minimum of this |ratio| will always yield a feasible solution:

$$
\min_{i|\beta_i>0}{\beta_i/\alpha_i}
$$

Back to matrix notation:

$$
\min_{i|A_B^{-1}A_E > 0}{(A_B^{-1}b)_i/(A_B^{-1}A_E)_i}
$$

Where we are doing element-wise divide over the two vectors, indexed by i (no sum).

```
L = idxmin(
        element_wise_divide(
            A_b.inv() @ A[:,E]),
            A_b.inv() @ b,
            return_null_if=A_b.inv() @ A[:,E]) <= 0
        )
)
```



<BR>
<BR>


# Dual Linear Program
## First Inuition
If we are maximizing $x_1 + 7 x_2$ subject to $x_1 + 7x_2 \le 15$ then we clearly have an upper bound for our objective (15). If we have can find a linear combination of constraints so that the coefficients of $x_i$ are larger than the coefficients in the objective, then the linear combination of the right handsides/constants of those constraints is an upper bound for the objective

## Introduction by Example
Maximize $x_1 + 10 x_2$ 

subject to

$$
\begin{aligned}
x_1 + 7 x_2 &\leq 17.5 \\
x_1 + 0 x_2 &\leq 3.5
\end{aligned}
$$

How do we find an upper bound for our obj?

Multiplying our constraints by $y_1$ and $y_2$ (and requiring they are both nonnegative) and then adding them we get

$$ (y_1 + y_2) x_1 + (7 y_1) x_2 \leq 17.5y_1 +3.5y_2   $$

if we have the constraints:

$$ 
\begin{aligned}
(y_1 + y_2) &\geq 1 (x_1 \text{'s coeff in the obj)} \\
7y_1 &\geq 10 (x_2 \text{'s coeff in the obj)}
\end{aligned} 
$$

Then by construction,

$$ 
x_1 + 10 x_2 \leq (y_1 + y_2) x_1 + (7 y_1) x_2 
\leq 17.5y_1 +3.5y_2 
$$

So to find the minimal lower bound for the maximization problem ($\max{x_1+x_10}$ subject to ...), we solve the problem:

Minimize $17.5y_1 + 3.5y_2 = b \cdot y$

Subject to:

$$
\begin{aligned}
    y_1 + y_2 &\geq 1 \\
    7 y_1 &\geq 10 \\
    (\Leftrightarrow
    A^T y &\geq c)
\end{aligned}
$$


Here we have a constructed a linear program and seen (and essentially proven) the weak duality theorem. 

From this example we can also see how each dual variable corresponds with a constraint.

It turns out theat when we solve the two programs, the two objectives coincide. This is called the strong duality theorem.

## General Rules
For any linear program of the form:

$$
\boxed{
    \begin{array}{cl}
        &\text{Max } c \cdot x \\
        &A x \leq b, x \geq 0 
    \end{array}
  }
$$

We can construct its dual program:

$$
\boxed{
    \begin{array}{rrr}
    % \begin{aligned}
        \text{min } b \cdot y \\
        \text{subj to } A^T y &\geq c \\
        y &\geq 0
    % \end{aligned}
    \end{array}
}
$$

We then have:

$$
\boxed{
    A \cdot x_{opt} = A^T \cdot y_{opt}    
}  
$$



# Dual Simplex
If we have an optimal basis and then we add a new variable to our LP, it is likely that the basis will no longer be optimal. But the BFS with the new variable = 0 will still be primal feasible, since the new variable won't contribute anything. 

If we add a new constraint to an LP, it is likely that the current BFS will violate this new constraint. However, adding a constraint to the primal LP is equivalent to adding a variable to the dual LP. So the corresponding dual's optimal solution will still be feasible (though probably not optimal). So we can dualize our LP and continue with the simplex method from the current dual solution until we reach dual optimality. Then we dualize back to our primal and we have found a primal feasible and optimal solution the LP with the new constraint. 


We can actually do this without explicitly creating the dual problem, but rather just analysing the primal tableau. Take the most negative basis variable to be the leaving variable:
```
L = min(
    A[:,basis].inv() @ b
)
```
<!-- 
take our tableau to be

given by

$$
\begin{aligned}
x_B &= A_B^{-1} b - A_B^{-1}A_N x_N\\
\\
&= 
\begin{array}{ccc}
+5 &+2x_1 &-1x_2 &+2x_3\\
-3 &+2x_1 &+4x_2 &-3x_3\\
-4 &+2x_1 &-3x_2 &+1x_3\\
\end{array} \\
z& = z - (1,3,2) \cdot x_N
\end{aligned}
$$

We pick the exiting variable to be one of the variables with a negative value in the solution - take the most negative, $x_5$. This is in fact equivalent to taking the variable with the most negative dual reduced costs to enter the dual solution, but let's ignore that for the moment. 

Now we need to pick an entering variable such that dual feasibility ( ie primal optimality) is maintained. 

(In the primal construction, this is equivalent to selecting the leaving variable, so it makes sense that we will need the ratio test).

So we have already decided to focus on row 3 ($x_5$ leaving) and we want to select an entering variable so that reduced costs remain positive and the constant in the row becomes positive. 

If we let $x_1$ or $x_6$ be the entering variable, when we rearrange row three, the basis value will be positive. Which do we choose though? We need to select the entering variable so that we retain our dual feasibility / primal optimality. 

Our two options are:

$$
\begin{aligned}
x_6 &= 4 - 2x_1 + 3x_3 + x_5 \text{ OR} \\
x_1 &= 2 +3/2 * x_3 + 1/2 * x_5 - 1/2 * x_6 \\
\end{aligned}
$$

We have

$$
z = 2 - x_1 -3x_3 -2x_6
$$

If we plug in the expression for $x_6$ (first option) into z, we see that we lose primal optimality as the coeffecient of $x_1$ would become positive.This is because 

$$
1 / 2 < 2 /1 \\
\text{red. costs}[1] / (-A_B^{-1}A_N [leaving:1]) < \text{red. costs}[6] / (-A_B^{-1}A_N [leaving:6])
$$

To see this rule in general, we can use the same kind of argument we did for the primal ratio test.  -->

Then we apply a ratio test similar to that in primal simplex. Let

$$
\begin{aligned}
x_L &= - \alpha + \beta_1 x_1 + \beta_2 x_2 - ... \text{(only interested in positive coefficients (ie when tableau is negative))} \\
z &= z_0 - rc_1 x_1 -rc_2 x_2 - ...(\text{all -ve since sol is  dual feasible})
\end{aligned}
$$

Let $x_1$ be the entering variable.

$$
\begin{aligned}
x_1 &= \alpha/\beta_1 + x_L/\beta_1 - \beta_2/\beta_1 * x_2 + ... \\
\Rightarrow
z &= z_0 - rc_1(\alpha/\beta_1 + x_L/\beta_1 - \beta_2/\beta_1 * x_2 + ...) - rc_2 x_2 - ...\\
&= z_0 - \alpha rc_1/\beta_1 - (rc_1/\beta_1)x_L  + (rc_1\beta_2/\beta_1  - rc2) x_2 \\
&= z_0 - \alpha rc_1/\beta_1 - (rc_1/\beta_1)x_L  + \beta_2(rc_1/\beta_1 - rc_2/\beta_2) x_2 \\
\end{aligned}
$$

Since all symbols are positive, looking at the coefficient of $x_2$ in our expression for z (and remembering primal optimal only if all coeffs in expr for z are negative) we can see that primal optimality / dual feasibility is retained only if

$$
rc_1 / \beta_1 < rc_2 \beta_2
$$

So, to find the leaving variable, $x_L$, we select the basis variable with the most negative value. To find the entering variable, we take

$$
\min_{i|\beta_i > 0} {rc_i / \beta_i}
$$

Where 

$$ x_L = \alpha - \sum_N \beta_N x_N \\
z = z_0 - \sum_N rc_N x_N
$$


In our matrix notation we have

$$
\begin{aligned}
(x_B)_L &= (A_B^{-1} b)_L - (A_B^{-1} A_N x_N)_L  \\
z &= z_0 - (c_B A_B^{-1} A_N - c_N) x_N \\
E &= \min_{i\vert(A_B^{-1} A_N )_{Li}<0}{\frac{ (c_B A_B^{-1} A_N - c_N)_i}{-(A_B^{-1} A_N)_{Li}}} \\
\end{aligned}
$$


```
E = min(
    element_wise_divide(
        (c_b @ A_b.inv() @ A_n - c_n),
        -A_b.inv[L,:] @ A_n,
        return_null_if=A_b.inv[L,:] @ A_n >= 0
    )
)
```

<BR>
<BR>

## Comparison with Dualizing and then doing normal simplex method
To understand this better, we can explicitly look at the dual problem at the same time and see how this matches up with the usual ratio test in the usual simplex algorithm.....
<BR>
<BR>
<BR>


# Integer Programming
## Branch and Bound
If we have an integer problem where some of the variables are required to be integral, we can use branch and bound with the simplex method. Branch and bound is a more general method/framework than just linear programming, but we don't need to explain the method in general.

- Solve the LP. If the solution is integral (as requred) then we are done. If not, then we take the relaxation value to be our upper bound and define a lower bound by finding an integral solution (eg round all variables down)
- Then:
    - we branch on one of the fractional variables, x1 with value v1 (for example pick the most fractional). We do this by creating two problems each being the original problem with a new constraint:
        - $v \leq \lfloor f \rfloor$ 
        - $v \geq \lceil f \rceil$
        - for each subproblem we calculate the relaxation value and the integral value. 
            - If the relaxation value is worse than our lower bound value, then we prune the branch - remove the node and all nodes below from the decision tree. 
            - if the integral value is better than our previous lower bound, then we can increase our lower bound score
    
- Continue with the above process, traversing more of the tree until our lower bound and upper bound are equal.

::: mermaid
graph TD;
    A{x1}-->|<= floor v1| B{x2}
    A{x1}-->|>=ceil v1| C{x2}
    B-->|<= floor v2|D{x3}
    B-->|>= ceil v2|E{x3}
    C-->|<= floor v2|F{x3}
    C-->|>= ceil v2|G{x3}
    D-->.
    E-->..
    F-->...
    G-->....
    ...
:::


## How do we do this in practise?
We don't want to actually build a tree, what we do instead is use a queue data-structure. 

<!-- Take the most fractional variable and branch - create two problems: -->
- Solve linear relaxation
- Queue = most fractional var, $x_{i_1}$

- Pop the first element of the queue, $x_{i_1}$ and 'branch' - create two problems:
    - LpProb & $x_{i_1} \geq \lceil v_{i_1} \rceil$ and LpProb & $x_{i_1} \leq \lfloor v_{i_1} \rfloor$
    - for each of these:
        - if relaxation < lower bound, do nothing (ie prune)
        - if integral sol > lower bound, replace lower bound
        - if relaxation > lower bound, then take the most fractional value, $x_{i_2}$, and add it to the queue queue.append(A,b,c, $x_{i_2}$). (A,b,c is the subproblem so it contains the information of the  of of the parents of the node). (This means we don't prune this branch, we continue it by picking the most fractional var to be the next node).


- Because we have developed the dual simplex method, that allows us to solve the branch problems quickly. On a branch node we have the basis, we then add the <= v1 or >= v1 constraint, making the problem primal infeasible. We use the dual simplex method to fix primal feasibility.

<!-- Take a maximization LP (```max c.x st A.x <= b```) with:
```
A = [4,1,0]
    [2,1,1]
    [1,0,1]

b = [6,4,2].T
c = [5,3,4].T
```

The optimal solution is ```x = [0.0, 2.0, 2.0]```

The dual is a minimization problem (```min c.y st A.y >= b```) with
```
A_dual = [4,2,1]
         [1,1,0]
         [0,1,1]

b_dual = [5,3,4].T
c _dual = [6,4,2].T
```

The optimal solution is ```y = [0.0, 3.0, 1.0]```


Let's now add a constraint to the original lp, so that the solution is infeasible.

```
A = [4,1,0]
    [2,1,1]
    [1,0,1]
    [0,1,1]

b = [5,3,4,3].T
```
$\Rightarrow$
```
A_dual = [4,2,1,0]
         [1,1,0,1]
         [0,1,1,1]
c_dual = [5,3,4,3].T
```

Our dual basis is ```[1,2]```, which remains dual feasible (since ```y[4]=0```). Then our reduced costs are:
``` 
RC = - c_dual[nonbasis] + c_dual[basis] @ A_dual[basis].inv() @A_dual[nonbasis] 
```
***
<BR><BR><BR>

|0|0|1|1|0|10| 
|-|-|-|-|-|-|
|1|0|2|-1|0|2|
|0|1|-1|1|0|2|
|0|0|-2|1|1|-1| 

Negative third RHS indicates infeasible primal third constraint $\leftrightarrow$ non optimal third dual variable. So $s_3$ / $x_5$ should be the **leaving variable**. If there are multiple, we can choose any - for example we may choose the smallest (most infeasible)

To fix infeasibility we want to make the negative RHS positive, so we pivot on a negative coefficented variable in the primal tableau. So in this case $s_1$ / $x_3$ is the **entering variable**. What if we have multiple negative coefficients? To choose which, remember we also need to maintain dual feasibility, this means reduced costs need to remain non-negative otherwise we may end up in a situation where we are primal infeasible and dual infeasible. Simplex cannot help in this situation. 

So we choose the non basic variable with the smallest
reduced costs / - coefficient -->