# Linear Programming
A linear program is a problem of the form

$$
\boxed {
\begin{array}{cc}
    \text{Maximize } c^T x \\
    \text{Subject to } Ax \leq b, x \geq 0 
\end{array}
}
$$

Actually they often look different but we can always convert them to look like this. 

To solve this problem we convert the inequality to an equality by adding slack variables. eg $x_1 \leq 1 \leftarrow x_1 + s_1 = 1$

The solution space to $Ax \leq b$ is called a bounded *convex*$^*$ polytope

<img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/3dpoly.svg" width=200>

## **Convex set**
We say a space S is convex if you can draw a line between any two points in the set, and every point on that line is in the set:

$$
\forall x,y \in S, \lambda \in [0,1] \Rightarrow\\
x + \lambda (y-x) \in S
$$

## **Intersection of convex sets is convex**
if $x,y \in S_1, S_2$ where $S_1$ and $S_2$ are both convex. Then for any $x,y \in S_1 \cap S_2, \lambda \in [0,1] \Rightarrow x + \lambda (y-x) \in S_1 \cap S_2$. This is useful because a half-plane: $\{x\in R^n | v \cdot x \leq b\}$ is convex, hence the intersection of many half-planes (the solution space to a linear program) is a convex set.

## **Optimal solution to an LP is a vertex of the solution polytope.**
An optimal solution will always sit on one of the vertices (red dots above). This is intuitively makes sense as we know that if you are optimizing a linear function, there will be no local maxima, so the maxima occurs at a boundary, and then we can apply the same logic to any boundary - the optimal solution along that boundary will be at the boundary of the boundary. And so on. So the optima will occur at the intersection of the maximum number of hyperplanes as possible. In 2d this is the intersection of two lines; in 3d this is the intersection of three planes. 

All of the above discussion was thinking in inequality land - how does this all work when we have converted to our equality version of the problem? 

Keeping with our inequalities for a moment, we conceptualise each constraint equation as a boundary/hyper-plane (solution is on the boundary when the inequality is tight) and we also have $x_1 \geq 0, x_2 \geq 0$ which similarly define boundaries. So we are at the intersection of D boundaries if D inequalities are tight. Now moving to our new formulation, this simply means D variables (including slacks and our original variables) are 0! So if we initially (before converting inequalities to equalities) have m constraints and n variables, then the solution will occur at the intersection of n of the m constraints. ie we will have n 0s in our optimal solution (when include the slack variables). This is called a **basic feasible solution**. 

## **Formal proof of the above**
### **A convex polytope is the convex combination of its vertices**
Take a point $p$ inside a polytope with vertices $\{v_P\}$. Take one of the vertices $v$ and consider where $\overrightarrow{vp}$ intersects with the boundary of the polytope, $B$. If it intesects at one the vertices, then we are done. Otherwise the point $k = \overrightarrow{vp} \cap B$ is interior to a $D - 1$ dimensional convex polytope. So we can prove this result inductively. A one dimensional polytope is just a line which clearly is a convex combination of its vertices. And we have just proved that if a $D - 1$ convex polytope is the convex combination of its vertices then a $D$ dimensional one is.
### **A convex function is globally optimal at a vertex**
Take a point, $p$, which is the unique global minumum of a linear function, $f$. Now by the above result:

$$
p = \sum{\lambda_i v_i} \hspace{0.1cm} \vert \hspace{0.1cm} \sum \lambda_i = 1
$$

By linearity of $f$,

$$
f(p) = \sum \lambda_i f(v_i)
$$

Since $f(p) \lt f(v_i) $ then
$$
\sum \lambda_i f(v_i) \gt \sum\lambda_i f(p) = f(p) \sum \lambda_i = f(p)
$$

Hence we have a contradiction.

## Basic feasible solution
A BFS is a vertex on a polytope. Equivalently, for a linear program $\max{c.x | A x = b, x \geq 0 }$, a BFS is 
a feasible solution x such that there exists an m element set 

$$
B \subset \{1, ..., n+m\}
$$

where

$$
A[:,B] \text{ is non singular}\\
j \notin B \Rightarrow x_j = 0 
$$

we call the variables $x_j, j \in B$ basic variables, and the rest nonbasic.

## Alternative definition of vertext
Unique maximum of linear function over polytope. 

...

## More intuition around a linear function's local maximum over a convex set being a global maximum
The simplex algorithm is a greedy algorithm that lets us move from one BFS / vertex to a better neighbouring BFS. This terminates when there are no better neighbouring BFSs. Although we are pretty confident (didn't prove it) that a solution occurs at a vertex, we are not sure that we can implement a greedy algorithm this way - eg imagine a set that has a kindof pointy snake a shape that snakes towards the optimum, away and then back towards the optimum. 

(We will now see that this cannot happen, because the intersection of linear inequalities cannot be snakey - it can only be kindof like a pointy ball. To see this formally:)

Let S be a concave set and let v be 'locally optimal' vertex. ie if you move in any direction from v, the objective does not get larger. Let s be another vertex that has a greater objective value.

$$
c \cdot s \gt c \cdot v  \\
\therefore \lambda \in [0,1] \Rightarrow \\
c \cdot (v + \lambda (s - v)) = \\
c \cdot v + \lambda(c \cdot s - c \cdot v) \gt c \cdot v
$$

Therefore, if we move at all toward s, the obj will increase - contradicting our assumption. The contradiction only exists if

$$ 
\lambda v + (1-\lambda)s \in S
$$

This holds true in any convex set by definition. If we have a snakey set, then this statement doesn't hold and in fact we can have locally optimal vertices that are not globally optimal. So what have we just proved? If we have a local maxima on our *convex* set and linear function, then it is a global maxima. This then implies that if a vertex has a larger objective than its neighbourhood (not neighbouring vertices) then it is the global maximum. 

Next thing would be to show that if larger than neighbourhood then larger than neighbouring vertices.

This definitely seems true geometrically. If you are at a vertice, any direction you move is a linear combination of the direction vectors towards the neighbouring vertices - therefore, bigger than neighbour vertices $\Rightarrow$ local maxima. Not sure how to show mathematically. 

## The Simplex Algorithm
From the above discussion, we have seen that an optimal solution to a linear program is on a vertex, and that if we keep moving to neighbouring vertices with larger objectives, we will eventually find the global max.

Take some basis B, then we have:

$$
\max{c_B^T x_B + c_N^T x_N}, \\
A_B x_B + A_N x_N = b, x \geq 0 
$$

Rearranging:

$$
\begin{aligned}
x_B &= A_B^{-1} (b - A_N x_N) = A_B^{-1}b \\
\text{reduced costs} &= c_B^T A_B^{-1}A_N − c_N^T \\
z &= c_B^T A_B^{-1}b - \text{reduced costs}   \cdot x_N,\\ x_N &= 0
\end{aligned}
$$


## Reduced costs $\rightarrow$ Entering variable
If the reduced cost of a non basic variable $x_E \in x_N$ is negative, then we can increase z by giving $c_N$ a positive value, so it 'enters the basis'. 

```
reduced_costs = c[basis] @ A[,:basis].inv() @ A[:,nonbasis] - c[nonbasis]
## getting actual idx, not iloc
E = idxmin(
    reduced_costs,
    return_null_if=reduced_costs.isnull()
)
```

## Is this well defined? What if we reorder the basis?
Let $\tilde{B}$ be a 'reordering' of $B$. Then we are reordering the columns of $c_B$ and $A_B$ by some permutation matrix $P$.

$$
c_{\tilde{B}} \cdot A_{\tilde{B}}^{-1} = (c_B \cdot P) \cdot (A_B \cdot P)^{-1} = c_B \cdot P P^{-1} A_B = c_B \cdot A_B
$$

So basically we have seen why the order of the columns in $A_B$ and $c_B$ is not important (but needs to be the same between the two).

## Interpretation of reduced costs
the reduced cost of variable j is by definition

$$
(\text{reduced cost})[j] = - \frac{\partial z}{\partial x_j}
$$

So it tells us what rate our objective will decrease by if we bring $x_j$ into the basis. 

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

## Example to illustrate the above
Take the following tableau:

$$
z = 6 - 3x_1 - 3x_2\\
x_3 = 1 - 3x_1 - 2x_2\\
x_4 = 2 - 2x_1 + 1x_2\\
x_5 = 3 + x_1 - 3x_1
$$

The negative cofficients in the expression for z (the positive reduced costs) show us that $z$ can be pushed down. Take $x_2$ to be the entering variable into the basis (make it non zero). Which variable shall we have leave the basis? We can't let it be $x_4$ as that would produce an infeasibility:
$$
x_2 = -2 + 2x_1 + x_4
$$

So it is between $x_1$ and $x_3$. Let's look at what happens in each option. Let $x_3$ be the leaving variable:

$$
\begin{aligned}
x_3 &= \frac{1}{2} - \frac{3}{2}x_1 - \frac{1}{2}x_3 \\ \Rightarrow
x_5 &= 3 + x_1  - 3(\frac{1}{2} +...)= \frac{3}{2} +...
\end{aligned}
$$

This is feasible because the constant / the coefficient the leaving variable * the coefficient of the leaving variable in the other equation is greater than 0. 

If we let $x_5$ be the leaving variable we would get:
$$
x_2 = 1 + ..\\
\Rightarrow x_3 = 1 - 3x_1 - 2(1 + ...) = -1 + ...
$$

Which is infeasible.s
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


## Simplex challenges
If we have a tableau like this:
$$
\begin{aligned}
z &= 6 - 3x_1 - 3x_2\\
x_3 &= 1 + 3x_1 - 2x_2\\
x_4 &= 3 + 2x_1 +1x_2
\end{aligned}
$$

In this case we 'can't pivot' when we take $x_1$ to be our leaving variable because there are no negative coefficients of $x_1$. What this means is that we can increase $x_1$ and arbitrarily decrease our cost function, the problem is unbounded.

## Cycling / Degeneracy

There are cases where selecting the largest reduced cost will lead to a pivot that does not improve the cost function (we are still on the same vertex because one of the basis variables is 0). This does not often happen but there are some ways around it - eg Bland's rule.


## Finding a first BFS
min $c.x$, st $A.x = b$.

Introduce artifical variables $y$.

$$
Ax + y = b
$$

Allow $y$ to be negative and set  $y = b$. Now:
Minimize $\sum y_i$, subject to $Ax + y = b$. There will be a feasible solution to the original problem if y=0 is the solution, and the result for $x$ can be taken as a solution to the initial problem.

# Dual Linear Program
## First Inuition
If we are maximizing $x_1 + 7 x_2$ subject to $x_1 + 7x_2 \le 15$ then we clearly have an upper bound for our objective (15). In general, if we have can find a linear combination of constraints so that the coefficients of $x_i$ are larger than the coefficients in the objective, then the linear combination of the right handsides/constants of those constraints is an upper bound for the objective

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

$$(y_1 + y_2) x_1 + (7 y_1) x_2 \leq 17.5y_1 +3.5y_2$$

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

We can do this same construction in general: take

$$
c^T x \\
\text{subject to } Ax \leq b
$$

Pre multiplying the constraint by y (row), where $y_i \geq 0$:

$$
y A x \geq y b
$$

If we constrain y by:

$$
y A \geq c^T
$$

Then by construction:

$$
yb \geq [yAx] \geq c^T x
$$

So to find the least upper bound for $c^T x$ we minimize $yb$ subject to $yA \geq c^T$.



It turns out that when we solve the two programs, the two objectives coincide. This is called the strong duality theorem.

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



# Dual Simplex
If we have an optimal basis and then we add a new variable to our LP, it is likely that the basis will no longer be optimal. But the BFS with the new variable = 0 will still be primal feasible, since the new variable won't contribute anything. 

If we add a new constraint to an LP, it is likely that the current BFS will violate this new constraint. However, adding a constraint to the primal LP is equivalent to adding a variable to the dual LP. So the corresponding dual's optimal solution will still be feasible (though probably not optimal). So we can dualize our LP and continue with the simplex method from the current dual solution until we reach dual optimality. Then we dualize back to our primal and we have found a primal feasible and optimal solution the LP with the new constraint. 


We can actually do this without explicitly creating the dual problem, but rather just analysing the primal tableau. Take the most negative basis variable to be the leaving variable:
```
L = min(
    A[:,basis].inv() @ b
)
```

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
rc_1 / \beta_1 \lt rc_2 \beta_2
$$

So, to find the leaving variable, $x_L$, we select the basis variable with the most negative value. To find the entering variable, we take

$$
\min_{i|\beta_i > 0} {rc_i / \beta_i}
$$

Where 

$$
x_L = \alpha - \sum_N \beta_N x_N\\
z = z_0 - \sum_N rc_N x_N
$$


In our matrix notation we have

$$
\begin{aligned}
(x_B)_L &= (A_B^{-1} b)_L - (A_B^{-1} A_N x_N)_L\\
z &= z_0 - (c_B A_B^{-1}A_N - c_N) x_N\\
E &= \min_{i|(A_B^{-1}A_N )_{Li} \lt 0} \frac{(c_B A_B^{-1}A_N - c_N)_i}{-(A_B^{-1}A_N)_{Li}}
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



## Comparison with dualizing and then doing normal simplex method
To understand this better, we can explicitly look at the dual problem at the same time and see how this matches up with the usual ratio test in the usual simplex algorithm.....

## Sufficiency for optimality
For a linear program and dual, with solutions $\tilde{x}, \tilde{y}$ respectively and $c^T \tilde{x} = \tilde{y} ^Tb$ then both solutions are optimal. 

**Proof:** by the weak duality theorem, $c^T x \leq y^T b$. Given $c^T \tilde{x} = \tilde{y}^T b$, then  $c^Tx \leq c^T \tilde{x}$ ie $\tilde{x}$ is optimal. We can use the exact same argument for y.


## Relationship between dual and primal solution
Let B be a basis for the primal optimal solution and let $y = c_B^T A_B^{-1}$. Then y is dual optimal.

$$
\begin{aligned}
y &= c_B A_B^{-1} \\
y^TA &= c_B^T A_B^{-1}A\\
 &= c_BA_B^{-1}[A_B, A_N] \\
&= [c_B, c_BA_B^{-1}A_N ] \\
\therefore
y^TA - c &= [0,  c_BA_B^{-1} - c_N ] \\
\text{[because B optimal]}&= [0, \text{reduced cost}] \geq 0\\
\therefore y^TA &\geq c \text{ y is feasible}
\end{aligned}
$$

and,

$$
y^Tb = c_B^TA_B^{-1}b = cB^Tx_B = c^T x
$$

By the sufficiency condition, y is optimal.
## Complementary slackness
Take primal problem: max $c^T x, Ax = b, x \geq 0$, and its corresponding dual $y^Tb, y^TA \geq c^T \leftrightarrow y^TA - v = c^T$. 

$$
\boxed{
\text{x, y optimal} \Leftrightarrow v^T x = 0
}
$$

**Proof** 

$$
\begin{aligned}
c^Tx &= (y^TA - v)x\\
&= y^TAx - vx\\
&= y^Tb - v^Tx\\
c^Tx = y^Tb &\Leftrightarrow v^Tx = 0
\end{aligned}
$$

# Integer Programming

## Gomory cuts

$$
x = A_B^{-1} b - A_B^{-1} A_N x_N\\
$$

The above is how we express each basic variable in terms of the non basic variables, giving us some representation of the constraints. In applying gomory cuts, we select a fractional basic variable and construct a new constraint out of the equation for that variable. 
$$
x_f = (A_B^{-1})_{fi} b_i - (A_B^{-1})_{fi} (A_N)_{ij} (x_N)_j :\Leftrightarrow: \\
x_f - \tilde{b}_f + \tilde{a}_{fj} x_j = 0
$$
(j summing over non basic).

Splitting this into fractional and integral parts, we get
$$
x_f - (\tilde{b}_f - \lfloor \tilde{b}_f \rfloor + \lfloor \tilde{b}_f \rfloor) 
+ (\tilde{a}_{fj} x_j  - \lfloor \tilde{a}_{fj} \rfloor x_j + \tilde{a}_{fj} x_j) = 0 \Rightarrow \\

x_f - \lfloor \tilde{b}_f \rfloor + \lfloor \tilde{a}_{fj} \rfloor x_j = \tilde{b}_f - \lfloor \tilde{b}_f \rfloor -( \tilde{a}_{fj} x_j - \lfloor \tilde{a}_{fj} \rfloor x_j )
$$

For any integer point inside the feasible region, the LHS is an integer. For any integer point inside the feasible region, the RHS is strictly less than 1. (first term is a fraction and second is negative). Therefore:

$
\tilde{b}_f - \lfloor \tilde{b}_f \rfloor -( \tilde{a}_{fj} x_j - \lfloor \tilde{a}_{fj} \rfloor x_j ) \leq 0
$

We have shown that the above inequality is true for any feasible integer point, however we have not shown any use of this fact. 

Our BFS has $x_j = 0$, so the RHS is strictly positive. Hence our BFS is removed from the region by adding the above constraint.

### Gomory Cut Recipe

1. Get relaxation sol
2. Find most fractional var:
```python
frac_var, frac_var_loc, frac_val = get_most_frac_var()
```
3. Get equation for that var:
```
a_gom = A_B_inv[frac_var_loc, :] @ A_N
b_gom = A_B_inv[frac_var_loc, :] @ b
```
4. New constraint is 
```
(a_gom - a_gom.floor()_ @ x[non_basis] + x[frac_var] <= b_gom - b_gom.floor()
```
Which is essentially
```
new_constraint = [
    a_gom[i] - floor(a_gom[i])
    if i in non_basis
    else 1 if i == frac_var
    else 0
    for i in vars
]
new_constraint @ x <= b_gom - floor(b_gom)
```

## Branch and bound
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