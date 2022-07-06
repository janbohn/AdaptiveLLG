# AdaptiveLLG
space and time adaptive algorithm for the LLG equation, preprint will follow shortly

Input:  Coarse mesh $\Tau^{-1}$ and fine time step size $\tau_{-1}$. Space tolerance $\tol_x$ and time tolerance $\tol_t$.   Error estimators in space $\err_x= \sum_{t\in\T}\err_x(T)$ and in time $\err_t$. Final time $T>0$. Parameters for the refinement and coarsening procedure $G_L<1$, $G_U>1$ and $0<\theta<1$ as percentage of too fine elements until coarsening happens.

1. Precomputation. 

as above 

2. Time stepping. 

Until $t_n<T$, do the following time stepping procedure: 


\begin{itemize}
\item Given: Approximation $m^{n}$ at time $t_{n}$ on mesh $\Tau^{n}$ and approximation $m^{n-1}$ at time $t_{n-1}$ on grid $\Tau^{n-1}$. Guess for $\tau_{n}$.
\item Start with guess for $\tau_{n}$ and $\Tau^{n+1}:=\Tau^n$, both are not accepted. 
\item As long as at least one of $\tau_{n}$ or $\Tau^{n+1}$ is not accepted: 
\begin{itemize} 
\item Interpolate $m^{n-1}$, $m^{n}$ on $\Tau^{n+1}$ (This is necessary due to (or simplifies a lot) the Fenics implementation). Proceed a time step with $\tau_{n}$ and mesh $\Tau^{n+1}$. %Compute approximation $m^{n+1}$ on $\Tau^n$ with $\tau_{n}$. 
\item If $\tau_{n}$ is not accepted: Compute error estimator $\err_t$ and guess for next time step size $\tau^*$. If $\err_t<\tol_t$ accept $\tau_{n}$ and use the guess for $\tau_{n+1}:=\tau^*$. Else use the guess for $\tau_{n}:=\tau^*$.
\item If $\Tau^{n+1}$ is not accepted: Compute error estimator $\err_x$ and refine all elements with $err_x(T)> G_U \tol_x $. Mark elements with $err_x(T)< G_L \tol_x $ for coarsening. 
\end{itemize}
\item If more than $\theta|\Tau^{n+1}|$ elements are marked for coarsening, coarsen the mesh, i.e.\ in the next time step use $\Tau^{n+1}:= \text{coarse}(\Tau^{n+1})$ as initial mesh. Compare Section~\ref{SSEC: Coarsening procedure} for the coarsening procedure.
\item Output: Approximation $m^n$ at time $t^n$ on mesh $\Tau^n$. Guess for $\tau_{n+1}$. Time step completed.
\end{itemize}


runs with FENICS 2019.1.0
