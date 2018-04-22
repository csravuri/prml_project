close all
clear all
clc

%% Data LOADING
load('/home/hadoop1/Documents/prml/project/matlab/dev_dish_washer.mat')
data_dw = vect;
load('/home/hadoop1/Documents/prml/project/matlab/dev_kettle.mat')
data_k = vect;
load('/home/hadoop1/Documents/prml/project/matlab/dev_running_machine.mat')
data_rm = double(vect);
load('/home/hadoop1/Documents/prml/project/matlab/dev_washing_machine.mat')
data_wm = vect;
b = [data_dw+data_k+data_rm+data_wm];

load('/home/hadoop1/Documents/prml/project/matlab/psi_dish_washer.mat')
Psi_dw = vect;
load('/home/hadoop1/Documents/prml/project/matlab/psi_kettle.mat')
Psi_k = vect;
load('/home/hadoop1/Documents/prml/project/matlab/psi_running_machine.mat')
Psi_rm = vect;
load('/home/hadoop1/Documents/prml/project/matlab/psi_washing_machine.mat')
Psi_wm = vect;

% load('/home/hadoop1/Documents/prml/project/UKDALE/dictionary20000_2itr/house2_dish_washer_dictionary.mat')
% Psi_dw = dish_washer{2};
% load('/home/hadoop1/Documents/prml/project/UKDALE/dictionary20000_2itr/house2_kettle_dictionary.mat')
% Psi_k = kettle{2};
% load('/home/hadoop1/Documents/prml/project/UKDALE/dictionary20000_2itr/house2_running_machine_dictionary.mat')
% Psi_rm = running_machine{2};
% load('/home/hadoop1/Documents/prml/project/UKDALE/dictionary20000_2itr/house2_washing_machine_dictionary.mat')
% Psi_wm = washing_machine{2};

A = [Psi_dw' Psi_k' Psi_rm' Psi_wm'];

n = size(A,2)

%% Specify DAY
day = 1;

for i = 1:48
    
    cvx_begin quiet
    variable x(n)
    minimize( norm(x,1) )
    subject to
        A * x == b(day,300*i-299:300*i)';
        -x<=0;
    cvx_end
    X(:,i) = x;
    
end

%% Plotting
estimate_dw = Psi_dw'*X(1:500,:);
estimate_dw = estimate_dw(:);
figure
plot(data_dw(day,:),'LineWidth',2)
hold on
plot(estimate_dw,'LineWidth',2);title('Dish Washer','FontSize',24)
xlabel('Sample','FontSize',24);
ylabel('Power (W)','FontSize',24);
lgd = legend('Original','Estimated');
lgd.FontSize = 24;
xt = get(gca, 'XTick');
set(gca, 'FontSize', 24)

estimate_k = Psi_k'*X(501:1000,:);
estimate_k = estimate_k(:);
figure
plot(data_k(day,:),'LineWidth',2)
hold on
plot(estimate_k,'LineWidth',2);title('Kettle','FontSize',24)
xlabel('Sample','FontSize',24);
ylabel('Power (W)','FontSize',24);
lgd = legend('Original','Estimated');
lgd.FontSize = 24;
xt = get(gca, 'XTick');
set(gca, 'FontSize', 24)

estimate_rm = Psi_rm'*X(1001:1500,:);
estimate_rm = estimate_rm(:);
figure
plot(data_rm(day,:),'LineWidth',2)
hold on
plot(estimate_rm,'LineWidth',2);title('Running Machine','FontSize',24)
ylabel('Power (W)','FontSize',24);
xlabel('Sample','FontSize',24);
lgd = legend('Original','Estimated');
lgd.FontSize = 24;
xt = get(gca, 'XTick');
set(gca, 'FontSize', 24)

estimate_wm = Psi_wm'*X(1501:2000,:);
estimate_wm = estimate_wm(:);
figure
plot(data_wm(day,:),'LineWidth',2)
hold on
plot(estimate_wm,'LineWidth',2);title('Washing Machine','FontSize',24)
ylabel('Power (W)','FontSize',24);
xlabel('Sample','FontSize',24);
lgd = legend('Original','Estimated');
lgd.FontSize = 24;
xt = get(gca, 'XTick');
set(gca, 'FontSize', 24)

figure
plot(b(day,:),'LineWidth',2)
hold on
plot(estimate_wm+estimate_k+estimate_rm+estimate_dw,'LineWidth',2);title('Aggregate','FontSize',24)
ylabel('Power (W)','FontSize',24);
xlabel('Sample','FontSize',24);
lgd = legend('Original','Estimated');
lgd.FontSize = 24;
xt = get(gca, 'XTick');
set(gca, 'FontSize', 24)



