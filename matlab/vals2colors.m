function[colors] = vals2colors(vals, cmap, res)
%VALS2COLORS project a vector of scalar values into (R,G,B) space
%
% Usage: colors = vals2colors(vals, [cmap], [res]);
%
% INPUTS:
%   vals: a vector of scalar values.  If vals is a matrix, it will be
%         re-shaped into a column vector.
%
%   cmap: an optional argument specifying (in a character string) which
%         colormap to use.  Default: 'linspecer'.
%
%    res: an optional argument specifying the maximum number of unique
%         colors to use.  Default: 100.
% 
% OUTPUTS:
% colors: a numel(vals) by 3 matrix of colors.
%
% SEE ALSO: PLOT_COORDS, LINSPECER
%
%  AUTHOR: Jeremy R. Manning
% CONTACT: jeremy.r.manning@dartmouth.edu

% CHANGELOG:
% 4-18-16  jrm  wrote it.


%convert values to RGB values using the specified colormap and resolution
if ~exist('res', 'var'), res = 100; end
if ~exist('cmap', 'var'), cmap = 'linspecer'; end

cmap = eval(sprintf('%s(%d)', cmap, round(res)));

vals = vals(:);
ranks = arrayfun(@(i)(sum(vals <= i)), vals);
ranks = round(res.*ranks./length(vals));
ranks(ranks < 1) = 1;

colors = cmap(ranks, :);