data1=readmatrix("trajectoriesMay22_12_26.txt");

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
axis([-5 5 -5 5])
grid on

L=10e-3; % Detector size (assume it's a square detector)
N=400; % Number of detectors (make sure it's a square)
l=L/N; % Make sure this returns an integer

S=linspace(-L/2,L/2,sqrt(N)); % Side of the detector (no need to create a meshgrid)
M=zeros(length(S));

for i=1:sqrt(N)-1
    for j=1:sqrt(N)-1
        for m=1:length(xf)
            if S(i)<=xf(m) && xf(m)<S(i+1) && S(j)<=yf(m) && yf(m)<S(j+1)
                M(i,j)=1;
            end
        end
    end
end

figure
[xx, yy] = meshgrid(1:size(M,1),1:size(M,2));
plot3(xx,yy,M,'*')