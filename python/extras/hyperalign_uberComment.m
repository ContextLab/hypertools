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
end

%hyperalign the given patterns.  assumption: all patterns have the same
%dimensionality.
dims = cellfun(@ndims, varargin);
assert(dims(1) == 2, 'trajectories must be specified in 2D matrices');

sizes = cellfun(@size, varargin, 'UniformOutput', false);
%trim rows to minimum number of rows
T = min(cellfun(@(s)(s(1)), sizes));
varargin = cellfun(@(x)(x(1:T, :)), varargin, 'UniformOutput', false);

%pad with zeros as needed
D = max(cellfun(@(s)(s(2)), sizes));
varargin = cellfun(@(x)([x zeros([size(x, 1) (D - size(x, 2))])]), varargin, 'UniformOutput', false);

sizes = cellfun(@size, varargin, 'UniformOutput', false);
template = sizes{1};
assert(all(cellfun(@(s)(all(s == template)), sizes)), 'all patterns must have same dimensionality');

%step 1: compute common template
for s = 1:length(varargin)        
    if s == 1
        template = varargin{s};
    else
        [~, next] = procrustes((template./(s - 1))', varargin{s}');
        template = template + next';
    end
end
template = template./length(varargin);

%step 2: align each pattern to the common template template and compute a
%new common template
template2 = zeros(size(template));
for s = 1:length(varargin)
    [~, next] = procrustes(template', varargin{s}');
    template2 = template2 + next';
end
template2 = template2./length(varargin);

%step 3: align each subject to the mean alignment from the previous round.
%save the transformation parameters
[aligned, transforms] = deal(cell(size(varargin)));
for s = 1:length(varargin)
    [~, next, transforms{s}] = procrustes(template2', varargin{s}');
    aligned{s} = next';
end
