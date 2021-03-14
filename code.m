clc;
close all;
clear all;


for k=0:9
for i=1:25
[x,fs]=audioread([num2str(k),'\',num2str(k),' (',num2str(i),').wav']);
x1=x(1000:end-1000);

y=abs(x1)>0.06;
p=find(y==1);
start=p(1);
endd=p(end);



seg=120;
vig=zeros(seg,16);
time=256;
sig=x1(start:1:start+seg*time-1);
%  plot(x1);hold on;stem([start,start+tsg-1],[2,2]);

% %%%% wavelet features ***********************

E=sum(sig.^2);
for w=1:seg


win=sig((w-1)*time+1:w*time);   


%%%%  energy ****************
e1=sum(win.^2);
qa=e1/E;

vig(w,1)=qa;

%%%% mfcc *******************
cx=(real(ifft(log(abs(fft(win,13))))))';
vig(w,2:14)=cx;

%%zero crossing ***********************

x1=[0;win];
x2=[win;0];

x3=x1.*x2;
x3=x3(2:end-1);
zc=sum(x3<=0);
zcf=zc/length(win);

vig(w,15)=zcf;

%%%%% entropy *********************
[coef,lg]=wavedec(win,5,'db1');
EN = wentropy(coef,'shannon');
vig(w,16)=EN;

end

F{k+1}{i}=vig;


end

end


%%%%%%%% Training *************************


train=[];
test=[];
group=[];
t=[];


u=size(F{1}{1},1);

%%%%% train data ************

for k=1:10
    for i=1:23
        
    train=[train;F{k}{i}];
    group=[group;(k-1)*ones(u,1)];
   
    end
    
end

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% classification and test 
%%%%% test data ****************************

for k=1:10
    for i=24:25
    
    test=[test;F{k}{i}];
   
    end
    
end

Class= knnclassify(test, train, group,1);
cl=reshape(Class,length(Class)/10,10);
out=zeros(10,2);

%%%%%***************  result ( out)   ***********

for i=1:10
    
    p=reshape(cl(:,i),u,2);
    
    for j=1:2
    h=hist(p(:,j),[0:9]);
    [b,n]=max(h);
    out(i,j)=n-1;
    end
end


display(out)





