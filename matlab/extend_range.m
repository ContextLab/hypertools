function[newlim] = extend_range(oldlim, prctile)

diff = oldlim(2) - oldlim(1);
newlim(1) = oldlim(1)-diff*prctile/100;
newlim(2) = oldlim(2)+diff*prctile/100;