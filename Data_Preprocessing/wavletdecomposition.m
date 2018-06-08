clc
clear all
close all
s=audioread('0.005_133_0.006_1.wav');
ls=length(s);
[C,L]=wavedec(s,5,'db4');
A5=wrcoef('a',C,L,'db4',5);
D1=wrcoef('d',C,L,'db4',1);
D2=wrcoef('d',C,L,'db4',2);
D3=wrcoef('d',C,L,'db4',3);
D4=wrcoef('d',C,L,'db4',4);
D5=wrcoef('d',C,L,'db4',5);

a5=sumsqr(A5);
d1=sumsqr(D1);
d2=sumsqr(D2);
d3=sumsqr(D3);
d4=sumsqr(D4);
d5=sumsqr(D5);

figure
subplot(611); plot(A5); title('A5');
subplot(612); plot(D1); title('D1');
subplot(613); plot(D2); title('D2');
subplot(614); plot(D3); title('D3');
subplot(615); plot(D4); title('D4');
subplot(616); plot(D5); title('D5');

