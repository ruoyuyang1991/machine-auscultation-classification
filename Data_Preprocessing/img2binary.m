clear all
clc
file_read=dir('/Volumes/RUOYU/mel-spectrom/speed_0_45_95/test/*.jpeg');
filenames={file_read.name}';
file_length=length(file_read);
M=zeros(22501,file_length);

for i=1:file_length  
    RGB=imread(strcat('/Volumes/RUOYU/mel-spectrom/speed_0_45_95/test/',file_read(i).name));
    G=rgb2gray(RGB);
    B=imbinarize(G);
    T=reshape(B,[22500,1]);
    N=[0;T];
    M(:,i)=N;
end
% % 
file_read1=dir('/Volumes/RUOYU/mel-spectrom/speed_1_133/test/*.jpeg');
filenames1={file_read1.name}';
file_length1=length(file_read1);
M1=zeros(22501,file_length1);
for i=1:file_length1  
    RGB1=imread(strcat('/Volumes/RUOYU/mel-spectrom/speed_1_133/test/',file_read1(i).name));
     G1=rgb2gray(RGB1);
     B1=imbinarize(G1);
    T1=reshape(B1,[22500,1]);
    N1=[1;T1];
    M1(:,i)=N1;
end
% % 
file_read2=dir('/Volumes/RUOYU/mel-spectrom/speed_2_190/test/*.jpeg');
filenames2={file_read2.name}';
file_length2=length(file_read2);
M2=zeros(22501,file_length2);
for i=1:file_length2  
    RGB2=imread(strcat('/Volumes/RUOYU/mel-spectrom/speed_2_190/test/',file_read2(i).name));
     G2=rgb2gray(RGB2);
     B2=imbinarize(G2);
    T2=reshape(B2,[22500,1]);
    N2=[2;T2];
    M2(:,i)=N2;
end
% % 
file_read3=dir('/Volumes/RUOYU/mel-spectrom/speed_3_256/test/*.jpeg');
filenames3={file_read3.name}';
file_length3=length(file_read3);
M3=zeros(22501,file_length3);
for i=1:file_length3  
    RGB3=imread(strcat('/Volumes/RUOYU/mel-spectrom/speed_3_256/test/',file_read3(i).name));
     G3=rgb2gray(RGB3);
     B3=imbinarize(G3);
    T3=reshape(B3,[22500,1]);
    N3=[3;T3];
    M3(:,i)=N3;
end

file_read4=dir('/Volumes/RUOYU/mel-spectrom/speed_4_375/test/*.jpeg');
filenames4={file_read4.name}';
file_length4=length(file_read4);
M4=zeros(22501,file_length4);
for i=1:file_length4  
    RGB4=imread(strcat('/Volumes/RUOYU/mel-spectrom/speed_4_375/test/',file_read4(i).name));
     G4=rgb2gray(RGB4);
     B4=imbinarize(G4);
    T4=reshape(B4,[22500,1]);
    N4=[4;T4];
    M4(:,i)=N4;
end

file_read5=dir('/Volumes/RUOYU/mel-spectrom/speed_5_530/test/*.jpeg');
filenames5={file_read5.name}';
file_length5=length(file_read5);
M5=zeros(22501,file_length5);
for i=1:file_length5  
    RGB5=imread(strcat('/Volumes/RUOYU/mel-spectrom/speed_5_530/test/',file_read5(i).name));
     G5=rgb2gray(RGB5);
     B5=imbinarize(G5);
    T5=reshape(B5,[22500,1]);
    N5=[5;T5];
    M5(:,i)=N5;
end

file_read6=dir('/Volumes/RUOYU/mel-spectrom/speed_6_750_1060/test/*.jpeg');
filenames6={file_read6.name}';
file_length6=length(file_read6);
M6=zeros(22501,file_length6);
for i=1:file_length6  
    RGB6=imread(strcat('/Volumes/RUOYU/mel-spectrom/speed_6_750_1060/test/',file_read6(i).name));
     G6=rgb2gray(RGB6);
     B6=imbinarize(G6);
    T6=reshape(B6,[22500,1]);
    N6=[6;T6];
    M6(:,i)=N6;
end

 M7=[M,M1,M2,M3,M4,M5,M6];
 r=randperm(size(M7,2));   
 BB=M7(:,r); 
 fid = fopen('/Volumes/RUOYU/mel-spectrom/speed_150_g_01_mel_bin/test_data.bin', 'w');
 fwrite(fid, BB, 'uint8');
 fclose(fid)
    