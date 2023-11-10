function plot_HW10_P4()
fsz = 20;
Na = 7; % the number of atoms
rstar = 2^(1/6);


[iter_BFGS,dxval_BFGS,xval_BFGS]=Rosenbrock_trust_region_BFGS_dogleg();
[iter_newton,dxval_newton,xval_newton]=Rosenbrock_trust_region();


%fval_final=[fval_sd(end),fval_newton(end),fval_BFGS(end),fval_FRCG(end),fval_PRCG(end)];
%gval_final=[gval_sd(end),gval_newton(end),gval_BFGS(end),gval_FRCG(end),gval_PRCG(end)];



figure(7);
clf;
hold on;
grid on;
plot(0:(iter_newton-1),dxval_newton(1:iter_newton),'Linewidth',3);
plot(0:(iter_BFGS-1),dxval_BFGS(1:iter_BFGS),'Linewidth',3);
legend('Newton','BFGS');
set(gca,'Fontsize',fsz);
set(gca, 'YScale', 'log');
title('Initial configuration=[-1.2,1.0], tol=1e^-6','FontSize',fsz)
xlabel('Iteration #','FontSize',fsz);
ylabel('||(x,y)-(x*,y*)||','FontSize',fsz);

figure(8);
X = linspace(-1.5,1.5);
Y = linspace(-1.5,1.5);
[X,Y] = meshgrid(X,Y,10);
Z = 100*(Y-X^2)^2+(1-X)^2;
hold on;
contour(X,Y,Z);
%legend('contour')
% the following line skip the name of the previous plot from the legend
%plot(xval_sd(1,:),xval_sd(2,:),'b-','LineWidth',2);
plot(xval_newton(1,:),xval_newton(2,:),'r-','LineWidth',2);
plot(xval_BFGS(1,:),xval_BFGS(2,:),'b-','LineWidth',2);
%plot(xval_FRCG(1,:),xval_FRCG(2,:),'m-','LineWidth',2);
%plot(xval_PRCG(1,:),xval_PRCG(2,:),'k-','LineWidth',2)
legend('contour','Newton','BFGS');

hold off
title('Trajectory of initial configuration=[-1.2,1.0], tol=1e^-6','FontSize',fsz)
xlabel('X','FontSize',fsz);
ylabel('Y','FontSize',fsz);


end


function xyz = initial_configuration(model,Na,rstar)
xyz = zeros(3,Na);
switch(model)
    case 1 % Pentagonal bipyramid
        p5 = 0.4*pi;
        he = sqrt(1 - (0.5/sin(0.5*p5))^2);
        for k = 1 : 5
            xyz(1,k) = cos((k-1)*p5); 
            xyz(2,k) = sin((k-1)*p5); 
            xyz(3,k) = 0;  
        end
        xyz(3,6) = he';
        xyz(3,7) = -he';

case 2 % Capped octahedron
        r = 1/sqrt(2);
        p4 = 0.5*pi;
        pp = p4/2;
        p0 = pi*1.5 - pp;
        x0 = sin(pp);
        y0 = cos(pp);
        z0 = 0;
        for k = 1 : 4
            xyz(1,k) = x0 + r*cos(p0 + (k-1)*p4);
            xyz(2,k) = y0 + r*sin(p0 + (k-1)*p4);
            xyz(3,k) = z0;
        end
        xyz(:,5) = [x0, y0, z0 + r]';
        xyz(:,6) = [x0, y0, z0 - r]';
        xyz(:,7) = [3*x0, y0, z0 + r]';

    case 3  % Tricapped tetrahedron
        p3 = 2*pi/3;
        pp = p3/2;
        r = 1/sqrt(3);
        beta = 0.5*pi -asin(1/3) - acos(1/sqrt(3));
        r1 = cos(beta);
        p0 = 1.5*pi - pp;
        x0 = sin(pp);
        y0 = cos(pp);
        z0 = 0;
        for k = 1 : 3
            xyz(1,k) = x0 + r*cos(p0 + (k-1)*p3);
            xyz(2,k) = y0 + r*sin(p0 + (k-1)*p3);
            xyz(3,k) = z0;
            xyz(1,k + 3) = x0 + r1*cos(p0 + pp + (k-1)*p3);
            xyz(2,k + 3) = y0 + r1*sin(p0 + pp + (k-1)*p3);
            xyz(3,k + 3) = z0 + sqrt(2/3) - sin(beta);
        end
        xyz(:,7) = [x0, y0, z0 + sqrt(2/3)]';

    case 4 % Bicapped trigonal bipyramid
        p3 = 2*pi/3;
        pp = p3/2;
        r = 1/sqrt(3);
        beta = 0.5*pi -asin(1/3) - acos(1/sqrt(3));
        r1 = cos(beta);
        p0 = 1.5*pi - pp;
        x0 = sin(pp);
        y0 = cos(pp);
        z0 = 0;
        for k = 1 : 3
            xyz(1,k) = x0 + r*cos(p0 + (k-1)*p3);
            xyz(2,k) = y0 + r*sin(p0 + (k-1)*p3);
            xyz(3,k) = z0;
        end
        xyz(:,4) = [x0 + r1*cos(p0 + pp), y0 + r1*sin(p0 + pp), z0 + sqrt(2/3) - sin(beta)]';
        xyz(:,5) = [x0 + r1*cos(p0 + pp + p3), y0 + r1*sin(p0 + pp+p3), z0 - sqrt(2/3) + sin(beta)]';
        xyz(:,6) = [x0, y0, z0 - sqrt(2/3)]';
        xyz(:,7) = [x0, y0, z0 + sqrt(2/3)]';

    otherwise % random configuration
        hR = 0.01;
        xyz = zeros(3,Na);
        xyz(:,1) = [0;0;0];
        a = randn(3,Na - 1);
        rad = sqrt(a(1,:).^2 + a(2,:).^2 + a(3,:).^2);
        a = a*diag(1./rad);
        for i = 2 : Na
            clear rad
            clear x
            rad = sqrt(xyz(1,1 : i - 1).^2 + xyz(2,1 : i - 1).^2 + xyz(3,1 : i - 1).^2);
            R = max(rad) + rstar;
            xa = R*a(:,i - 1);
            x = [xyz(:,1 : i - 1), xa];
            f = LJ(x(:));
            fnew = f;
            while 1
                R = R - hR;
                xa = R*a(:,i - 1);
                x = [xyz(:,1 : i - 1), xa];
                f = fnew;
                fnew = LJ(x(:));
                if fnew > f
                    break;
                end
            end
            xyz(:,i) = xa;
        end
        cmass = mean(xyz,2);
        xyz = xyz - cmass*ones(1,Na);
end
xyz = xyz*rstar;
end

function v = LJ(xyz) % 
global feval
feval = feval + 1;
m = length(xyz);
Na = m/3;
x = reshape(xyz,3,Na);
r = zeros(Na);
for k = 1 : Na
    r(k,:) = sqrt(sum((x - x(:,k)*ones(1,Na)).^2,1));
end
r = r + diag(ones(Na,1));
aux = 1./r.^6;
L = (aux - 1).*aux;
L = L - diag(diag(L));
v = 2*sum(sum(L));
end