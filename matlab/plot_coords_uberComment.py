#FULL BREAKDOWN OF CONCEPTUAL STEPS

#Testing with weights.m 
#cell array (one element per subject); 
#each element is timepoints by brain regions matrix
#(story-listening data)


#color_specified = false;
#^??

#if iscell(x)
#k: if an array (1)

#    h = zeros(size(x));
#    k: make same size array, zeros

#    args = cell(size(varargin));
#    k: create cell array, size of number of inputs (varargin=1xN array)

#    for i = 1:length(varargin)        
#         if iscell(varargin{i})
#    	  k: for each cell in varargin


#             assert(all(size(x) == size(varargin{i})), 'All cell array arguments must be the same size');
#			  k: 


#             if any(strcmpi('color', varargin{i}))
#			  color_specified = true;
#             k: if the string 'color' is in varargin cell (strcmpi=1), then color is specified

#             end

#             args{i} = varargin{i};
#             k: args becomes a cell-for-cell copy of varargin{i}?

#         else

#             if strcmpi('color', varargin{i})
#                 color_specified = true;
#			  k: redundant?

#             end

#             next = cell(size(x));
#			  k: next =empty cells same dimension as x

#             [next{:}] = deal(varargin{i});
#			   each argument, n, in varargins becomes a cell in next?

#             args{i} = next;
# 			  next becomes cell in args

#         end            

#     end

#     if ~color_specified

#         colors = linspecer(numel(x));

#     end

#     h_state = ishold;

#     if ~h_state, cla; end

#     hold on;

#     for i = 1:numel(x)

#         if ~color_specified

#             color_args = {'Color', colors(i, :)};

#         else

#             color_args = {};

#         end

#         next_args = cellfun(@(x)(x{i}), args, 'UniformOutput', false);

#         if ~isempty(next_args)

#             c = parse_plot_color(next_args{1}); %one more possible check...

#             if ~isempty(c)

#                 color_args = {};

#             end

#         end

#         h(i) = main_helper(x{i}, next_args{:}, color_args{:});        

#     end

#     if ~h_state, hold off; end

# else

#     h = main_helper(x, varargin{:});

# end



# function[h] = main_helper(x, varargin)

# if isempty(varargin) && size(x, 2) >= 2, varargin = {'k.'}; end



# if size(x, 2) == 2

#     h = plot(x(:, 1), x(:, 2), varargin{:});

# elseif size(x, 2) == 3

#     h = plot_coords_3d_helper(x, varargin{:});

# elseif size(x, 2) > 3    

#     %do PCA on x (don't require stats toolbox; compute with SVD)

#     %center the data first...

#     x = x - repmat(nanmean(x, 1), [size(x, 1) 1]);



#     %remove all nans

#     bad_inds = sum(isnan(x), 2) > 0;
#     clean_x = x(~bad_inds, :);

#     if isempty(clean_x)

#         h = [];

#         return;

#     end

#     [~, ~, v] = svd(clean_x, 'econ');

#     score = x*v;

#     h = plot_coords_3d_helper(score(:, 1:3), varargin{:});

# elseif size(x, 2) == 1

#     h = bar(x, varargin{:});

# end

    

# function[h] = plot_coords_3d_helper(x, varargin)    

# if ismatrix(varargin{1}) && (size(varargin{1}, 2) == 3) && (size(varargin{1}, 1) == size(x, 1))

#     colors = varargin{1};    

#     S = 10;    

#     h = scatter3(x(:, 1), x(:, 2), x(:, 3), S, colors, 'filled');  

#     grid off;

# else

#     h = plot3(x(:, 1), x(:, 2), x(:, 3), varargin{:});

# end

# set(gca, 'projection', 'perspective');




# function[color] = parse_plot_color(s)

# colors = {'b' 'g' 'r' 'c' 'm' 'y' 'k' 'w'};

# markers = {'.' 'o' 'x' '+' '*' 's' 'd' 'v' '^' '<' '>' 'p' 'h' 'square' 'diamond' 'pentagram' 'hexagram'};

# linestyles = {'-' ':' '-.' '--'};



# %remove linestyle and markers.  is the only thing left a colorstring?

# s = remove_substrings(s, markers);

# s = remove_substrings(s, linestyles);

# if any(strcmp(s, colors))

#     color = s;

# else

#     color = [];

# end






# function[y] = remove_substrings(s, strs)

# remove_inds = false(size(s));

# for i = 1:length(strs)    

#     next = strfind(s, strs{i});

#     for j = 1:length(next)

#         remove_inds(next:(next+length(s)-1)) = true;

#     end

# end

# y = s(~remove_inds);


