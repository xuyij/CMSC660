function plot_final_Q2()
fsz = 20;

kmax=10000;

[gnorm,residual]=final_nesterov(kmax);


%fval_final=[fval_sd(end),fval_newton(end),fval_BFGS(end),fval_FRCG(end),fval_PRCG(end)];
%gval_final=[gval_sd(end),gval_newton(end),gval_BFGS(end),gval_FRCG(end),gval_PRCG(end)];



figure(7);
clf;
hold on;
grid on;
plot(1:kmax,gnorm(1:kmax),'Linewidth',3);
legend('Newton');
set(gca,'Fontsize',fsz);
set(gca, 'YScale', 'log');
title('norm of gradient','FontSize',fsz)
xlabel('Iteration #','FontSize',fsz);
ylabel('||g||','FontSize',fsz);

figure(8);
clf;
hold on;
grid on;
plot(1:kmax,residual(1:kmax),'Linewidth',3);
legend('Newton');
set(gca,'Fontsize',fsz);
set(gca, 'YScale', 'log');
title('norm of residual','FontSize',fsz)
xlabel('Iteration #','FontSize',fsz);
ylabel('||r||','FontSize',fsz);


end
