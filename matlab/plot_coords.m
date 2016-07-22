function[h] = plot_coords(x, varargin)
%PLOT_COORDS Plot a series of coordinates in 1, 2, or 3d
%
% Usage: h = plot_coords(X, [colors], ...)
%
% INPUTS:
%
%      X: a T by D matrix of observations.  T is the number of coordinates
%         and D is the dimensionality of each observation.  NaNs are
%         treated as missing observations.
%
% colors: an optional argument specifying the colors to make each point.
%         Only valid if D > 1.  This can be the output of vals2colors.
% 
% Any additional input arguments are passed into bar (if D == 1), plot (if
% D == 2), or plot3 (if D >= 3).  if the colors are specified, the
% remaining arguments are ignored.
%
% OUTPUTS:
%     h: handle to the lineseries object.
%
% NOTE:
% X can also be a cell array, which causes plot_coords to be called for
% each element of X.  Subsequent arguments may either be cell arrays (of
% the same size) or non-cell arrays (e.g. strings, matrices).  If X is a
% cell array, subsequent cell array arguments will be used to parameterize
% the corresponding plot for each element of X.  If subsequent arguments
% are not cell arrays, the same parameters (for that argument) will be used
% to plot every element of X.  If no colors are specified, each trajectory
% or set of points (i.e., elements of X) is plotted in a different color.
%
% EXAMPLES:
%
% %Plot a 10-D random walk in 3-D
% x1 = cumsum(randn(1000, 10), 1);
% plot_coords(x1, 'k-', 'LineWidth', 2);
%
% %Generate a 4-D random walk.  Plot the first 3 dimensions and color the
% %points according to the 4th dimension.
% x2 = cumsum(randn(1000, 4), 1);
% plot_coords(x2(:, 1:3), vals2colors(x2(:, 4)), 'o');
%
% %plot 2 5-D Gaussian point clouds.
% x3 = 2.*randn(1000, 5);
% x4 = randn(100, 5);
% plot_coords({x3 x4}, {'o' 's'}, 'MarkerSize', {5 10}, 'LineWidth', 2);
%
% SEE ALSO: PLOT, PLOT3, SCATTER, SCATTER3, BAR, VALS2COLORS,
%           TRAJECTORY_PLOTTER
%
%  AUTHOR: Jeremy R. Manning
% CONTACT: jeremy.r.manning@dartmouth.edu

% CHANGELOG:
% 4-18-16  jrm  wrote it.
% 4-20-16  jrm  support cell arrays

color_specified = false;
if iscell(x)
    h = zeros(size(x));
    args = cell(size(varargin));
    for i = 1:length(varargin)        
        if iscell(varargin{i})
            assert(all(size(x) == size(varargin{i})), 'All cell array arguments must be the same size');
            if any(strcmpi('color', varargin{i}))
                color_specified = true;
            end
            args{i} = varargin{i};
        else
            if strcmpi('color', varargin{i})
                color_specified = true;
            end
            next = cell(size(x));
            [next{:}] = deal(varargin{i});
            args{i} = next;
        end            
    end
    if ~color_specified
        colors = linspecer(numel(x));
    end
    h_state = ishold;
    if ~h_state, cla; end
    hold on;
    for i = 1:numel(x)
        if ~color_specified
            color_args = {'Color', colors(i, :)};
        else
            color_args = {};
        end
        next_args = cellfun(@(x)(x{i}), args, 'UniformOutput', false);
        if ~isempty(next_args)
            c = parse_plot_color(next_args{1}); %one more possible check...
            if ~isempty(c)
                color_args = {};
            end
        end
        h(i) = main_helper(x{i}, next_args{:}, color_args{:});        
    end
    if ~h_state, hold off; end
else
    h = main_helper(x, varargin{:});
end


function[h] = main_helper(x, varargin)
if isempty(varargin) && size(x, 2) >= 2, varargin = {'k.'}; end

if size(x, 2) == 2
    h = plot(x(:, 1), x(:, 2), varargin{:});
elseif size(x, 2) == 3
    h = plot_coords_3d_helper(x, varargin{:});
elseif size(x, 2) > 3    
    %do PCA on x (don't require stats toolbox; compute with SVD)
    %center the data first...
    x = x - repmat(nanmean(x, 1), [size(x, 1) 1]);

    %remove all nans
    bad_inds = sum(isnan(x), 2) > 0;
    clean_x = x(~bad_inds, :);
    if isempty(clean_x)
        h = [];
        return;
    end
    [~, ~, v] = svd(clean_x, 'econ');
    score = x*v;
    h = plot_coords_3d_helper(score(:, 1:3), varargin{:});
elseif size(x, 2) == 1
    h = bar(x, varargin{:});
end


function[h] = plot_coords_3d_helper(x, varargin)    
if ismatrix(varargin{1}) && (size(varargin{1}, 2) == 3) && (size(varargin{1}, 1) == size(x, 1))
    colors = varargin{1};    
    S = 10;    
    h = scatter3(x(:, 1), x(:, 2), x(:, 3), S, colors, 'filled');  
    grid off;
else
    h = plot3(x(:, 1), x(:, 2), x(:, 3), varargin{:});
end
set(gca, 'projection', 'perspective');


function[color] = parse_plot_color(s)
colors = {'b' 'g' 'r' 'c' 'm' 'y' 'k' 'w'};
markers = {'.' 'o' 'x' '+' '*' 's' 'd' 'v' '^' '<' '>' 'p' 'h' 'square' 'diamond' 'pentagram' 'hexagram'};
linestyles = {'-' ':' '-.' '--'};

%remove linestyle and markers.  is the only thing left a colorstring?
s = remove_substrings(s, markers);
s = remove_substrings(s, linestyles);
if any(strcmp(s, colors))
    color = s;
else
    color = [];
end


function[y] = remove_substrings(s, strs)
remove_inds = false(size(s));
for i = 1:length(strs)    
    next = strfind(s, strs{i});
    for j = 1:length(next)
        remove_inds(next:(next+length(s)-1)) = true;
    end
end
y = s(~remove_inds);


