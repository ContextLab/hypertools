function[] = trajectory_plotter(varargin)
%TRAJECTORY_PLOTTER  Beautifully animate a series of high-D trajectories
%
% Description: Project the coordinates x1, ...., xn onto 3 dimensions using
% PCA.  Then create a movie where a sliding window of coordinates is
% plotted for each frame.
%
% Usage: trajectory_plotter([outfile], W, x1, x2, x3, ..., xN)
%
% INPUTS:
%                  outfile: an optional argument specifying the filename of
%                           a movie of the animation.  If ommitted, no
%                           movie is created (instead the animation is
%                           displayed).
%
%                        W: a scalar integer specifying the number of
%                           timepoints to display in each frame.
%
%      x1, x2, x3, ..., xN: number-of-observations by number-of-dimensions
%                           coordinate matrices.  All matrices must have
%                           the same number of dimensions, and the number
%                           of dimensions must be >= 2.  If more than 3
%                           dimensions are used, trajectory_plotter uses
%                           PCA to project onto 3 dimensions.  Each
%                           trajectory is plotted in a different color.
%
% EXAMPLE:
%
% %plot 3 5-D random walks, each in a different color.  Save the output to
% 'trajectories_test.avi' in the current working directory.
% x1 = cumsum(randn(1000, 5), 1);
% x2 = cumsum(randn(1000, 5), 1);
% x3 = cumsum(randn(1000, 5), 1);
% trajectory_plotter('trajectories_test', 10, x1, x2, x3);
%
% SEE ALSO: PLOT_COORDS
%
%  AUTHOR: Jeremy R. Manning
% CONTACT: jeremy.r.manning@dartmouth.edu

% CHANGELOG:
% 4-18-16  jrm  wrote it.
% 6-29-16  jrm  bug fixes.

%TRAJECTORY_PLOTTER
%
%Usage: trajectory_plotter([outfile], windowlength, x1, x2, ..., xn)
%
%
%Project the coordinates x1, ...., xn onto 3 dimensions using PCA.  Then
%create a movie where a sliding window of coordinates is plotted for each
%frame.
assert(length(varargin) >= 2, 'Usage: trajectory_plotter([outfile], W, x1, x2, ..., xn)');

if ischar(varargin{1})
    outfile = varargin{1};
    varargin = varargin(2:end);
else
    outfile = '';
end

windowlength = varargin{1};
varargin = varargin(2:end);

nfeatures = cellfun(@(x)(size(x, 2)), varargin);
nfeatures = nfeatures(nfeatures ~= 0);
assert(length(unique(nfeatures)) == 1, 'All coordinates must have same number of dimensions');
assert(nfeatures(1) >= 2, 'Must use at least 2 dimensions');

colors = linspecer(length(varargin));
inds = [];
coords = [];
for i = 1:length(varargin)
    coords = [coords ; varargin{i}];
    inds = [inds ; i.*ones([size(varargin{i}, 1) 1])];
end

%plot parameters
D = min(3, size(coords, 2));
EXPAND_RES = 3;
FOV = 75;

%do PCA on x (don't require stats toolbox; compute with SVD)
%center the data first...
coords = coords - repmat(nanmean(coords, 1), [size(coords, 1) 1]);
bad_inds = sum(isnan(coords), 2) > 0; %remove all nans
clean_coords = coords(~bad_inds, :);
if isempty(clean_coords), return; end
[~, ~, v] = svd(clean_coords, 'econ');
score = coords*v;
score = score(:, 1:D);

smooth_score = [];
smooth_inds = [];
for i = 1:length(varargin)
    next_inds = find((inds == i));
    X = 1:length(next_inds);
    XX = linspace(1, length(next_inds), EXPAND_RES*length(next_inds));
    next_smoothed = pchip(X, score(next_inds, :)', XX)';
    smooth_score = [smooth_score ; next_smoothed]; %#ok<*AGROW>
    smooth_inds = [smooth_inds ; i.*ones([size(next_smoothed, 1) 1])];
end

if ~isempty(outfile)
    vidObj = VideoWriter(outfile);
    vidObj.FrameRate = 15*EXPAND_RES;
    vidObj.Quality = 100;
    open(vidObj);
end

n_timepoints = arrayfun(@(i)(sum(smooth_inds == i)), 1:length(varargin));
limits = [min(smooth_score) ; max(smooth_score)];
u = linspace(0, 2*pi, min(n_timepoints));
m = mean(smooth_score, 1);
r = 1.2*pdist2(smooth_score, m, 'Euclidean', 'Largest', 1);

fignum = figure('units', 'pixels', 'position', [0 0 800 600]);
for i = 1:(min(n_timepoints) - windowlength*EXPAND_RES + 1)
    clf;
    hold on;
    for j = 1:length(varargin)
        next_inds = find((smooth_inds == j));
        c = lighten(colors(j, :), 0.9);
        plot_coords(smooth_score(next_inds(1:i), 1:D), ':', 'LineWidth', 1, 'Color', c);        
    end
    for j = 1:length(varargin)
        next_inds = find((smooth_inds == j));        
        plot_coords(smooth_score(next_inds(i:(i+windowlength*EXPAND_RES-1)), 1:D), '-', 'LineWidth', 4, 'Color', colors(j, :));        
    end
    hold off;
    box on; grid on;
    set(gca, 'XTickLabel', [], 'YTickLabel', [], 'ZTickLabel', [], 'LineWidth', 2, 'XTick', [], 'YTick', [], 'ZTick', []);
    xlim(extend_range([limits(1, 1) limits(2, 1)], 50));
    ylim(extend_range([limits(1, 2) limits(2, 2)], 50));
    if size(limits, 2) >= 3
        zlim(extend_range([limits(1, 3) limits(2, 3)], 50));
        camtarget(m);
        campos(m + [r*cos(u(i)) r*sin(u(i)) 0]);
        camup([0 0 1]);        
        camva(FOV);
    end
    
    if ~isempty(outfile)
        f = getframe;
        writeVideo(vidObj, f);
    else
        drawnow;
    end
end
if ~isempty(outfile)
    close(vidObj);
end
close(fignum);
