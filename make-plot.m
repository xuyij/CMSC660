function make_plot()

error_CG=random_walk_in_maze();
error_precond_CG=precond_random_walk_in_maze();
semilogy(1:size(error_CG,2),error_CG,1:size(error_precond_CG,2),error_precond_CG,'LineWidth',3);
xlabel('iterations');
ylabel('error (norm of residue)'); 
legend('Conjugate Gradient Descend','Preconditioned Conjugate Gradient Descend')
end
