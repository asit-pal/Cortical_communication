%% Load and pre-process neural data

load 107l003p143.mat

BIN_WIDTH = DRIVEN_TRIAL_LENGTH; % in ms.
[spikes, stim] ...
    = ExtractSpikes(neuralData, BIN_WIDTH, 'TrialPeriod', 'Driven');
% The 'TrialPeriod' option can be set to 'Driven' (trials for which an
% oriented grating was presented), 'Spontaneous' (trials for which no
% grating was presented), 'Full' (combine each driven trial with the
% subsequent spontaneous trial), or an interval in ms (interval must be in
% the range 1 - 2780 (the first 1280 ms correspond to driven activity, the
% subsequent 1500s correspond to spontaneous activity)

%% Apply PCA

X = permute( spikes{V1}, [3 1 2] );
% X is a number of trials by number of V1 units matrix, where the (i, j)
% entry indicates the number of spikes fired by unit j on trial i.

V = pca(X);
Z = X*V(:,1:3);

%% Plot 3D projection onto the top three principal directions. Color
% changes with grating orientation. Each point corresponds to a different
% trial.

load COLOR_MAP
COLOR_MAP = COLOR_MAP(1:8:end,:);

stimIds = unique(stim);
numStim = numel(stimIds);

figure('visible', 'off')  % Create figure without displaying it
hold on
for stimIdx = 1:numStim
    
    plot3( ...
        Z(stim == stimIds(stimIdx),1), ...
        Z(stim == stimIds(stimIdx),2), ...
        Z(stim == stimIds(stimIdx),3), ...
        'ko', 'MarkerFaceColor', COLOR_MAP(stimIdx,:), ...
        'MarkerSize', 12)
    
end
hold off

axis off
axis equal

% Save the figure
saveas(gcf, 'Figures/pca_plot_driven.png')
disp('Plot saved as pca_plot_driven.png')

