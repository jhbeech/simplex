# Simplex Algorithm

Maximize
$c^T x = c_B^T x_B + c_N^T x_N$

Subject to
$$
\begin{aligned}
Ax &= b, \text{ ie} \\
A_B x_B + A_N x_N &= b, x \geq 0 
\end{aligned}
$$

## Method
Pick a basis $\rightarrow$ $x_B, x_N, z$ $\rightarrow$ reduced costs $\rightarrow$ entering & leaving variable $\rightarrow$ new basis


$$
\begin{aligned}
x_B &= A_B^{-1} (b - A_N xN) = A_B^{-1}b \\
\text{reduced costs} &= c_B^T A_B^{-1}A_N − c_N^T \\
z &= c_B^T A_B^{-1}b
\end{aligned}
$$
