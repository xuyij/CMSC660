function [mag_list]=Ising_MC(beta,kmax)


% initialize spin configurations
N=30;
config=2*randi([0,1],N)-1;
mag_list=zeros([kmax,1]);


for iter=1:kmax
%randomly choose a spin
    i=randi([1,30]);
    j=randi([1,30]);
    dE=config(i,j);
    
    dE=2*config(i,j)*(config(mod(i,N)+1,mod(j-1,N)+1) ...
        +config(mod(i-2,N)+1,mod(j-1,N)+1) ...
        +config(mod(i-1,N)+1,mod(j,N)+1) ...
        +config(mod(i-1,N)+1, mod(j-2,N)+1));
    
    alpha=min(exp(-beta*dE),1);
    seed=rand;

    if seed<alpha
        config(i,j)=-config(i,j);
    end
    mag=sum(config,"all")/(N^2);
    mag_list(iter)=mag;

    
end

%figure(1);
%clf;
%hold on;
%grid on;
%plot(1:kmax,mag_list,'Linewidth',3);
%title('magnetization','FontSize',20)
%xlabel('Iteration #','FontSize',20);
%ylabel('magnetization','FontSize',20);

end


