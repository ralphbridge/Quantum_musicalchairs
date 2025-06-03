data1=readmatrix("trajectoriesMay22_12_26.txt");

N=100;
xi=zeros(length(data1)/2,1);
xf=zeros(length(data1)/2,1);
yi=zeros(length(data1)/2,1);
yf=zeros(length(data1)/2,1);
zi=zeros(length(data1)/2,1);
zf=zeros(length(data1)/2,1);

for i=1:length(xi)
    xi(i)=data1(2*i-1,1);
    xf(i)=data1(2*i,1);
    yi(i)=data1(2*i-1,2);
    yf(i)=data1(2*i,2);
    zi(i)=data1(2*i-1,3);
    zf(i)=data1(2*i,3);
end

figure
plot(xi*1e3,yi*1e3,'.')
hold on
plot(xf*1e3,yf*1e3,'.')
xlabel('x mm','FontSize',16)
ylabel('y mm','FontSize',16)