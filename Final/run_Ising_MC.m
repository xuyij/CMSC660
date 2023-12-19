function run_Ising_MC()


kmax=10^7;

inv_temp= 0.2:0.01:1;
mean_mag=zeros([size(inv_temp,2),1]);
var_mag=zeros([size(inv_temp,2),1]);



for i=1:size(inv_temp,2)
    mag=Ising_MC(inv_temp(i),kmax);
    mean_mag(i)=mean(mag);
    var_mag(i)=var(mag);
        
    
end

mu=zeros(size(inv_temp));
mu(26:81)=(1-sinh(2*inv_temp(26:81)).^(-4)).^(1/8);
mu(1:25)=0;

figure(1);
clf;
hold on;
grid on;
plot(inv_temp,abs(mean_mag),'Linewidth',3);
plot(inv_temp,abs(mean_mag)+sqrt(var_mag),'Linewidth',3);
plot(inv_temp,abs(mean_mag)-sqrt(var_mag),'Linewidth',3);
plot(inv_temp,mu,'Linewidth',3);
title('magnetization','FontSize',20)
xlabel('beta','FontSize',20);
ylabel('magnetization','FontSize',20);
legend('<m>','<m>+var(m)','<m>-var(m)','exact')

end
