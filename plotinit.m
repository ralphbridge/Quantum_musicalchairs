str='trajectoriesAug01_00_05.txt';
data=readmatrix(str);
N=size(data,1)/3;
xi=zeros(N,1);
yi=zeros(N,1);
zi=zeros(N,1);

xf=zeros(N,1);
yf=zeros(N,1);
zf=zeros(N,1);

vx1=zeros(N,1);
vy1=zeros(N,1);
vz1=zeros(N,1);

vx2=zeros(N,1);
vy2=zeros(N,1);
vz2=zeros(N,1);

ux=zeros(N,2);
uy=zeros(N,2);
uz=zeros(N,2);

relx=zeros(N,1);
rely=zeros(N,1);
relz=zeros(N,1);
for i=1:N
    xi(i)=data(3*i-2,2);
    yi(i)=data(3*i-2,3);
    zi(i)=data(3*i-2,4);
    xf(i)=data(3*i,2);
    yf(i)=data(3*i,3);
    zf(i)=data(3*i,4);
    vx1(i)=data(3*i-1,5);
    vy1(i)=data(3*i-1,6);
    vz1(i)=data(3*i-1,7);
    vx2(i)=data(3*i,5);
    vy2(i)=data(3*i,6);
    vz2(i)=data(3*i,7);
    ux(i,1)=vx1(i)/(vx1(i)+vy1(i)+vz1(i));
    uy(i,1)=vy1(i)/(vx1(i)+vy1(i)+vz1(i));
    uz(i,1)=vz1(i)/(vx1(i)+vy1(i)+vz1(i));
    ux(i,2)=vx2(i)/(vx2(i)+vy2(i)+vz2(i));
    uy(i,2)=vy2(i)/(vx2(i)+vy2(i)+vz2(i));
    uz(i,2)=vz2(i)/(vx2(i)+vy2(i)+vz2(i));
    relx(i)=1-(ux(i,2)/ux(i,1));
    rely(i)=1-(uy(i,2)/uy(i,1));
    relz(i)=1-(uz(i,2)/uz(i,1));
end

%% Plotting section

figure
subplot(3,1,1)
plot(relx,'.')
title(str,'interpreter','none','FontSize',20)
ylabel('$\Delta\hat{u}_x$','interpreter','latex','FontSize',20)
subplot(3,1,2)
plot(rely,'.')
ylabel('$\Delta\hat{u}_y$','interpreter','latex','FontSize',20)
subplot(3,1,3)
plot(relz,'.')
ylabel('$\Delta\hat{u}_z$','interpreter','latex','FontSize',20)
xlabel('$N$','interpreter','latex','FontSize',20)

figure
plot3(xi,yi,zi,'.')
axis equal;
title(['Initial distribution for N=',num2str(N),' electrons'])

figure
plot3(xf,yf,zf,'.')
axis equal;
title(['Distribution at detector position for N=',num2str(N),' electrons'])