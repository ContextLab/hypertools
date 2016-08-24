function[aligned, transforms] = hyperalign(varargin)
%HYPERALIGN  Hyperalign a series of high-dimensional trajectories
%
%This function implements the "hyperalignment" algorithm described by the
%following paper:
%
%  Haxby JV, Guntupalli JS, Connolly AC, Halchenko YO, Conroy BR, Gobbini
%  MI, Hanke M, and Ramadge PJ (2011)  A common, high-dimensional model of
%  the representational space in human ventral temporal cortex.  Neuron 72,
%  404 -- 416.
%
%Hyperalignment computes the affine transformations (rotations,
%reflections, and scalings) that brings the trajectories into best
%alignment
%
% NOTE: The Statistics Toolbox is needed to use this function.
%
%Usage:
% [aligned, transforms] = hyperalign(x1, x2, x3, ..., xN)
%
%INPUTS:
%      x1, x2, x3, ..., xN: number-of-observations by number-of-dimensions
%                           coordinate matrices.  Across all coordinate
%                           matrices, the minimum number of observations
%                           will be considered, and all coordinate matrices
%                           are assumed to start at the same time.  If some
%                           observations have fewer dimensions, they will
%                           be padded with zeros to ensure that all
%                           observations have the same dimensionality.
%
%OUTPUTS:
%                 aligned: a 1 by N cell array of the hyperaligned
%                          trajectories.
%
%              transforms: a 1 by N cell array of transformation matrices
%                          that were applied to the inputs to produce the
%                          aligned trajectories.  Each transform is a
%                          struct with the following fields:
%                             c: the translation component
%                             T: the orthogonal rotation and reflection
%                                component
%                             b: the scale componenet
%
%EXAMPLE:
%
% %create 5 25-dimensional random walks.  Hyperalign them and then use
% %trajectory_plotter to visualize them.
% walks = arrayfun(@(x)(cumsum(randn(500, 25), 1)), 1:5, 'UniformOutput', false);
% aligned = hyperalign(walks{:});
% trajectory_plotter(10, aligned{:});
%
% SEE ALSO: PROCRUSTES, TRAJECTORY_PLOTTER
%
%  AUTHOR: Jeremy R. Manning
% CONTACT: jeremy.r.manning@dartmouth.edu

% CHANGELOG:
% 4-18-16  jrm  wrote it.
% 4-20-16  jrm  be more forgiving of mismatched sizes.

assert(license('test','Statistics_toolbox') == 1, ...
    'You must install the Statistics Toolbox to use this function');

if length(varargin) <= 1
    aligned = varargin;
    return;
    %if there is only one input, it does not need to be aligned (there is
    %nothing to align to but itself)
end

%hyperalign the given patterns.  assumption: all patterns have the same
%dimensionality.
dims = cellfun(@ndims, varargin);
%outputs 1x36 vector, each value the number of dimensions of the original
%data
%cellfun applies function to each cell of an array
%we obtain the number of dimensions of each array in the input arguments


assert(dims(1) == 2, 'trajectories must be specified in 2D matrices');
%the first matrix must be be 2d, so we assert the first value of dims is 2
%assert gives error if condition is false

sizes = cellfun(@size, varargin, 'UniformOutput', false);
%outputs an array of 2-number arrays that represent the dimensions
%trim rows to minimum number of rows
%options 'UniformOutput', 'false' combines the outputs into cell arrays 
T = min(cellfun(@(s)(s(1)), sizes));
%min- smallest elements in an array 
%CLARIFY: I think this is taking the smallest element from the list of all
%first elements in the sizes arrays?
varargin = cellfun(@(x)(x(1:T, :)), varargin, 'UniformOutput', false);
%cuts the inputs so that they all have the same number of rows (equal to
%the smallest number of rows)

%pad with zeros as needed
D = max(cellfun(@(s)(s(2)), sizes));
%find the max number of columns (taking the max element from the list of all
%second elements in the sizes array)
varargin = cellfun(@(x)([x zeros([size(x, 1) (D - size(x, 2))])]), varargin, 'UniformOutput', false);
%all are made to match the largest number of columns
%need to fill in extra columns? fills in with zeros

sizes = cellfun(@size, varargin, 'UniformOutput', false);
%options 'UniformOutput', 'false' combines the outputs into cell arrays 
template = sizes{1};
%vector with values of the first array in sizes (size of first input
%argument dimensions)
assert(all(cellfun(@(s)(all(s == template)), sizes)), 'all patterns must have same dimensionality');
%make sure all in sizes have been comformed to the same size (same as
%template)

%step 1: compute common template
for s = 1:length(varargin)        
    if s == 1
        template = varargin{s};
        %assign template to first input argument values 
        
    else
        %for the 'next' input? 
        [~, next] = procrustes((template./(s - 1))', varargin{s}');
        %determines linear transformation of points in transposed varargin{s} matrix
    %to best conform them to the transpose of (template./(s-1))
        template = template + next';
        %keep adding the transpose of the results to template
        %so, in the end, each value in template matrix is a sum of all the
        %procrustes outputs for that cell
    end
end
template = template./length(varargin);
%divide each value by the number of input arguments


%step 2: align each pattern to the common template and compute a
%new common template
template2 = zeros(size(template));
for s = 1:length(varargin)
    [~, next] = procrustes(template', varargin{s}');
    %use the transpose of the template you just generated to generate a new
    %template which matches the transpose of the input argument to the
    %transpose of the first template
    
    %QUESTION: why transpose?
    
    %determines linear transformation of points in the varargin{s} matrix
    %to best conform them to the transpose of the template
    template2 = template2 + next';    
    %this output becomes the template for the next transformation?
end
template2 = template2./length(varargin);
%divide each value in template2 by the number of input arguments

%step 3: align each subject to the mean alignment from the previous round.
%save the transformation parameters
[aligned, transforms] = deal(cell(size(varargin)));
for s = 1:length(varargin)
    [~, next, transforms{s}] = procrustes(template2', varargin{s}');
    aligned{s} = next';
end
