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

%expect numpy.matrix or numpy.array


color_specified = false;
if iscell(x)
%if an array
    h = zeros(size(x));
    %row vector, zeros, length size(x)
    args = cell(size(varargin));
    %create one cell for each additional input arg in array
    for i = 1:length(varargin)
        if iscell(varargin{i})
        %for each cell in varargin
            assert(all(size(x) == size(varargin{i})), 'All cell array arguments must be the same size');
            %errors if size of weights array != soze of each cell in
            %varargsin array
            if any(strcmpi('color', varargin{i}))
                color_specified = true;
                %if the string 'color' is in any varargin cell, then color
                %is specified
            end
            args{i} = varargin{i};
            %CLARIFY: args becomes a cell for cell copy of varargsin?
        else
            if strcmpi('color', varargin{i})
                color_specified = true;
                %if varargin not an array, just see if color is specified
            end
            next = cell(size(x));
            %make cell array, size(x)
            [next{:}] = deal(varargin{i});
            %ALL next cells are varargin{i}
            args{i} = next;
            %next becomes a cell in args
            %as we cycle through the for loop, each cell in args
            %becomes a 1x36 array, with 36 copies of original varargin
        end            
    end
    if ~color_specified
        colors = linspecer(numel(x));
        %CLARIFY: gives colors for each x
    end
    h_state = ishold;
    %holds plot so contents can be added
    if ~h_state, cla; end
    %clear axes
    hold on;
    %keep new axes from erasing existing plot
    for i = 1:numel(x)
        %numel- # of array elements 
        if ~color_specified
            color_args = {'Color', colors(i, :)};
        else
            color_args = {};
        end
        %if color assignments exist, add them to color_args
		%if not, make empty array color_args

        next_args = cellfun(@(x)(x{i}), args, 'UniformOutput', false);
        %cellfun applies function to each cell of an array
		%seems to copy the first subj from weights into first two cells of next_args array
            %CLARIFY: maybe to compare to the two varargins (used two in test run)?
        if ~isempty(next_args)
            c = parse_plot_color(next_args{1}); %one more possible check...
            %looks again for color, just in case
			%see parse_plot_color function, below
            if ~isempty(c)
                color_args = {};
                %still nothing found? color_args --> empty array
            end
        end
        h(i) = main_helper(x{i}, next_args{:}, color_args{:});        
        %see main_helper function, below
    end
    if ~h_state, hold off; end
    %end holding the plot
else
    %if x not an array, feed into main_helper function
    h = main_helper(x, varargin{:});
end


function[h] = main_helper(x, varargin)
if isempty(varargin) && size(x, 2) >= 2, varargin = {'k.'}; end
    %if no additional inputs, and columns >=2
    %if varargsin empy, just use k
if size(x, 2) == 2
    h = plot(x(:, 1), x(:, 2), varargin{:});
    %if 2 columns, plot col1 x col2 and use varargin{:}
elseif size(x, 2) == 3
    h = plot_coords_3d_helper(x, varargin{:});
    %if 3 columns, plot x with vararhgin using 3d_helper 
elseif size(x, 2) > 3    
    %do PCA on x (don't require stats toolbox; compute with SVD)
        %principle component analysis
    %center the data first...
    x = x - repmat(nanmean(x, 1), [size(x, 1) 1]);
      %nanmean(x,1)- column averages, ignoring NaNs
      %[size(x, 1) 1]- rows in x, 1
      %repmat- outputs matrix same size as x with column values all equal to the nanmean of that column in x
      %center data around zero by subtracting the mean from each value
    
    %remove all nans
    bad_inds = sum(isnan(x), 2) > 0;
    %bad_inds= rows by 1 matrix, with 1 representing rows in x containing NaNs, zeros are "good" rows
    clean_x = x(~bad_inds, :);
    %clean_x is the set of all rows not contatining NaNs
    
    if isempty(clean_x)
        h = [];
        %if there are no rows containing to NaNs, declare h=[]
        return;
    end
    [~, ~, v] = svd(clean_x, 'econ');
        %help svd-  maybe only v
    score = x*v;
    %score/factor loading/weight matrix - score here is weight matrix
    %csikit learn package &/or numpy look for pca
    h = plot_coords_3d_helper(score(:, 1:3), varargin{:});
   
    %#QUESTION: SIMPLE EXPLANATION OF SINGLE VALUE DECOMPOSITION? (videos suck)
		%[U,S,V] = svd(X) produces a diagonal matrix S, of the same 
        %dimension as X and with nonnegative diagonal elements in
        %decreasing order, and unitary matrices U and V so that
    	%X = U*S*V'.
        
elseif size(x, 2) == 1
    h = bar(x, varargin{:});
    %one dimension? make a bar graph!
end


function[h] = plot_coords_3d_helper(x, varargin)    
if ismatrix(varargin{1}) && (size(varargin{1}, 2) == 3) && (size(varargin{1}, 1) == size(x, 1))
    %if varargin{1} is a matrix AND 3 columns in first cell of varargin AND
    %rows in varargsin = rows in x
    colors = varargin{1}; 
    %set colors
    S = 10;    
    %CLARIFY: what's this?
    h = scatter3(x(:, 1), x(:, 2), x(:, 3), S, colors, 'filled');  
    %3d scatterplot of the three columns in x, marker size=10, colors
    %assigned from varargsin, filled circles
    grid off;
    %no ugly grid lines, thanks
else
    h = plot3(x(:, 1), x(:, 2), x(:, 3), varargin{:});
    %if varargsin not a matrix, just feed varargsin into function as individual
    %arguments
end
set(gca, 'projection', 'perspective');
%slightly changes the view/visualizaiton/perspective somehow?


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
%matrix of zeros, size s
for i = 1:length(strs)    
    next = strfind(s, strs{i});
    %for each in strs, search s for that string
    %next = list of all concurrent strings
    for j = 1:length(next)
        %for each in next
        remove_inds(next:(next+length(s)-1)) = true;
        %CLARIFY
    end
end
y = s(~remove_inds);


