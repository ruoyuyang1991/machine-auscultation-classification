clear all
clc
% a=csvread('/Users/ruoyuyang/Downloads/run_logtrain_k5_2c2m-tag-accuracy_accuracy_accuracy.csv',1,1);
% b=csvread('/Users/ruoyuyang/Downloads/run_logvaln_k5_2c2m-tag-accuracy_accuracy_accuracy.csv',1,1);
c=csvread('/Users/ruoyuyang/Downloads/run_logtrain_k5_2c2m_drop0.8_40000-tag-loss_loss_loss.csv',1,1);
d=csvread('/Users/ruoyuyang/Downloads/run_logvaln__k5_2c2m_drop0.8_40000-tag-loss_loss_loss.csv',1,1);
% output1=medfilt1(a(:,2),2);
% output2=medfilt1(b(:,2),2);
output3=medfilt1(c(:,2),5);
output4=medfilt1(d(:,2),5);

figure(1)
plot(c(:,1),output3);
hold
plot(d(:,1),output4);
legend('training data','validation data');
xlabel('training step')
ylabel('loss')
ylim([0,1])
title('Convergence of Chatter function loss')