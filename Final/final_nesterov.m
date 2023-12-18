function [gnorm,residual]=final_nesterov(kmax)
close all
fsz = 20;
N=91;

%% Initialization
x=normrnd(0,N,[N,1]);
y=normrnd(0,N,[N,1]);
A = readmatrix("Adjacency_matrix.csv");
%% optimize w and b 

% params for SINewton
%bsz = 50;
%kmax = 1e3;
tol = 1e-3;
iter=1;
f=zeros(kmax);
gnorm=zeros(kmax);
residual=zeros(kmax);
lam=0.01;
gamma=0.9;
c=0.1;

alpha=0.05;
lr=0.05;


%
while iter<kmax 
    if iter==1
        %Ig = randperm(n,bsz);
        g=-forces(x,y,A);
        if norm(g)>1
            g=g/norm(g);
        end
        p=-g;
        a=1.0;
        
        xold=x;
        yold=y;
        x = x + lr*p(1:N);
        y = y + lr*p(N+1:2*N);
    else
        %fprintf('success!')
        mu=1-3/(5+iter);
        tx=(1+mu)*x-mu*xold;
        ty=(1+mu)*y-mu*yold;
        g=-forces(tx,ty,A);
        if norm(g)>1
            g=g/norm(g);
        end
        p=-g;
        a=1.0;
        ftemp=potential(tx+a*p(1:N),ty+a*p(N+1:2*N),A);
        if ftemp>potential(x,y,A)
            a=a*gamma;
            if a < 1e-14
                fprintf("line search failed\n");
                iter = iter_max;
                fail_flag = 1;
                break;
            end
            
        end
        xold=x;
        yold=y;
        x=tx+a*p(1:N);
        y=ty+a*p(N+1:2*N);
    end
    
  
    
    %w=w-alpha*g;
    f(iter)=potential(x,y,A);
    gnorm(iter)=norm(forces(x,y,A));
    residual(iter)=norm(p+forces(xold,yold,A));
    iter=iter+1;


   
 
end


%[w,f,gnorm] = LevenbergMarquardt(r_and_J,w,kmax,tol);
plot_graph(x,y,A);

end
%%
%%
function f = forces(x,y,A)
% f = force = - grad U = column vector with 2*N components
% x, y are column vectors with N components
% A is an N-by-N adjacency matrix
N = length(x);
%% find pairwise distances between linked vertices
xaux = x*ones(size(x))';
yaux = y*ones(size(y))';
dx = A.*xaux - A.*(xaux'); 
dy = A.*yaux - A.*(yaux');
dxy = sqrt(dx.^2+dy.^2);

%% spring forces due to linked vertices
Aind = find(A == 1);
idiff = zeros(N);
idiff(Aind) = 1 - 1./dxy(Aind);
fx = -sum(idiff.*dx,2);
afx = min(abs(fx),1);
sfx = sign(fx);
fx = afx.*sfx;

fy = -sum(idiff.*dy,2);
afy = min(abs(fy),1);
sfy = sign(fy);
fy = afy.*sfy;

f_linked = [fx;fy];

%% repelling spring forces due to unlinked vertices
h = sqrt(3);
Aind = find(A==0);
A = ones(size(A))-A;
dx = A.*xaux - A.*(xaux'); 
dy = A.*yaux - A.*(yaux');
dxy = sqrt(dx.^2+dy.^2);
fac = zeros(N);
diff = dxy - h;
fac(Aind) = min(diff(Aind),0); 
fx = sum(fac.*dx,2);
fy = sum(fac.*dy,2);
f_unlinked = -[fx;fy];

f = f_linked + f_unlinked;
end
%%%%%%%%%%%%%%%%%%%

function f = potential(x,y,A)
% f = force = - grad U = column vector with 2*N components
% x, y are column vectors with N components
% A is an N-by-N adjacency matrix
N = length(x);
%% find pairwise distances between linked vertices
xaux = x*ones(size(x))';
yaux = y*ones(size(y))';
dx = A.*xaux - A.*(xaux'); 
dy = A.*yaux - A.*(yaux');
dxy = sqrt(dx.^2+dy.^2);

%% spring forces due to linked vertices
Aind = find(A == 1);
idiff = zeros(N);
idiff(Aind) = (dxy(Aind)-1).*(dxy(Aind)-1);
linked_energy=1/2*sum(idiff,"all");

%% repelling spring forces due to unlinked vertices
h = sqrt(3);
Aind = find(A==0);
diff(Aind) = (dxy(Aind)-sqrt(3)).*(dxy(Aind)-sqrt(3));
unlinked_energy=1/2*sum(diff,"all");

f = linked_energy + unlinked_energy;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_graph(x,y,A)
figure;
hold on
plot(x,y,'o','Markersize',15,'MarkerEdgeColor',[0.5,0,0],'MarkerFaceColor',[1,0,0]);
ind = find(A == 1);
[I,J] = ind2sub(size(A),ind);
for k = 1 : length(ind)
    plot([x(I(k)),x(J(k))],[y(I(k)),y(J(k))],'linewidth',4,'Color',[0,0,0.5]);
end
daspect([1,1,1])
axis off
end





