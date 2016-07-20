function[c] = lighten(c, amt)
c = rgb(c); %accept character strings, etc.

if ~exist('amt', 'var'), amt = 0.5; end

c = amt*([1 1 1]) + (1 - amt)*c;
