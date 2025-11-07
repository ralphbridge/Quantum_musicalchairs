clc
clear all

str='Oct14_00_28';
data=readmatrix(['trajectories',str,'.txt']);

N=1000;
expected_rows=3*N;

for i=1:length(data)
    data(i,12)=data(i,12)+1;
end
%% Finding missing rows and replacing them
particleID = data(:,12);

% Count number of rows per particle
counts = accumarray(particleID, 1, [N 1]);

% Find which particles are missing rows
missingParticles = find(counts < 3);

% Report
fprintf('Total rows in file: %d (expected %d)\n', size(data,1), expected_rows);
fprintf('Number of particles with missing rows: %d\n', length(missingParticles));

if ~isempty(missingParticles)
    disp('Particles with missing rows:');
    disp(missingParticles(:)');
end

% --- Fill missing rows ---
numCols = size(data,2);

for p = missingParticles'
    missingCount = 3 - counts(p);
    % Create placeholder rows: [NaN ... NaN, particleID]
    newRows = [NaN(missingCount, numCols)];
    newRows(:,12) = p;  % put particle ID in column 12
    newRows(:,13)=10000;
    % Append to data
    data = [data; newRows];
end

% Sort rows by particle ID (so they stay grouped)
data = sortrows(data, [12 13]);

%% Filtering out particles that didn't make it to the detector

xi=zeros(length(data)/3,1);
xf=zeros(length(data)/3,1);
yi=zeros(length(data)/3,1);
yf=zeros(length(data)/3,1);
zi=zeros(length(data)/3,1);
zf=zeros(length(data)/3,1);

for i=1:length(xi)
    xi(i)=data(3*i-2,2);
    xf(i)=data(3*i,2);
    yi(i)=data(3*i-2,3);
    yf(i)=data(3*i,3);
    zi(i)=data(3*i-2,4);
    zf(i)=data(3*i,4);
end

% figure
% plot(xi*1e3,yi*1e3,'.')
% hold on
% plot(xf*1e3,yf*1e3,'.')
% xlabel('x mm','FontSize',16)
% ylabel('y mm','FontSize',16)
% axis([-6 6 -6 6])
% xTickLocations=-6:1:6;
% yTickLocations=-6:1:6;
% set(gca,'XTick',xTickLocations,'YTick', yTickLocations);
% axis square;
% grid on

L=10e-3; % Detector size (assume it's a square detector)
N=100; % Number of detectors (make sure it's a square)
l=L/N; % Make sure this returns an integer

for i=length(missingParticles):-1:1
    xi(missingParticles(i))=[];
    xf(missingParticles(i))=[];
    yi(missingParticles(i))=[];
    yf(missingParticles(i))=[];
    zi(missingParticles(i))=[];
    zf(missingParticles(i))=[];
end

S=linspace(-L/2,L/2,sqrt(N)+1); % Side of each detector
M=zeros(length(S)-1);
for i=1:sqrt(N)
    for j=1:sqrt(N)
        for m=1:length(xf)
            if S(i)<=xf(m) && xf(m)<S(i+1) && S(j)<=yf(m) && yf(m)<S(j+1) && data(3*m,13)<100000
                M(sqrt(N)+1-j,i)=1;
            end
        end
    end
end

writematrix(M,['detector',str,'.csv'])
disp(str);