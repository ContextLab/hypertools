% rgb.m: translates a colour from multiple formats into matlab colour format
% type 'rgb demo' to get started
%
% [matlabcolor]=rgb(col)
% matlab colors are in the format [R G B]
%
% if 'col' is a string, it is interpreted as
%
% 	[[modifier] descriptor] colour_name
%
% where
%		modifier is one of   (slightly, normal, very, extremely)
%		descriptor is one of (light/pale, normal, dark)
%		colorname is a name of a colour
%			(type 'rgb list' or 'rgb demo' to see them all)
%
% if 'col' is an integer between 0 and &HFFFFFF inclusive,
% it is interpreted as a double word RGB value in the form
% [0][R][G][B]
%
% if 'col' is a negative integer between -1 and -&HFFFFFF
% inclusive, it is interpreted as the complement of a double
% word RGB value in the form [0][B][G][R]
%
% if 'col' is a string of the form 'qbX' or 'qbXX' where X
% is a digit then the number part is interpreted as a qbasic
% color
%
% if 'col' is one of {k,w,r,g,b,y,m,c} a sensible result is
% returned
%
% if 'col' is already in matlab format, it is unchanged

%	VERSION:	06/06/2002
%	AUTHOR:		ben mitch
%	CONTACT:	footballpitch@theconnexion.zzn.com
%	WWW:		www.benmitch.co.uk\matlab (not yet)
%	LOCATION:	figures\colors\


% Copyright (c) 2009, Ben Mitch
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.

function out=rgb(in)

if isa(in,'char') & length(in)>2 & length(in)<5 & strcmpi('qb',in(1:2))
	out=qbcolor(sscanf(in(3:end),'%i'));
elseif isa(in,'char') & length(in)==1
	out=translatecolorchar(in);
elseif isa(in,'char')
	if strcmp(in,'demo') rgb_demo; return; end
	if strcmp(in,'list') rgb_list; return; end
	out=translatecolorstring(in);
elseif isa(in,'double') & size(in,1)==1 & size(in,2)==1 & abs(in)<16777216
	out=translatecolorRGB(in);
elseif isa(in,'double') & size(in,1)==1 & size(in,2)==3
	out=in;
else
	warning('Unrecognised color format, black assumed');
	out=[0 0 0];
end

function out=translatecolorchar(in)
switch(in)
case 'k', out=[0 0 0];
case 'w', out=[1 1 1];
case 'r', out=[1 0 0];
case 'g', out=[0 1 0];
case 'b', out=[0 0 1];
case 'y', out=[1 1 0];
case 'm', out=[1 0 1];
case 'c', out=[0 1 1];
otherwise
	warning(['Unrecognised colour "' in '", black assumed'])
	out=[0 0 0];
	return;
end

function out=translatecolorstring(in)
args.tokens=rgb_parse(in);
args.N=length(args.tokens);
if args.N>3 warning('Too many words in color description, any more than 3 will be ignored'); end
while(args.N<3)
	args.tokens=[{'normal'};args.tokens];
	args.N=args.N+1;
end

cols=get_cols;
col=[];
for n=1:size(cols,1)
	names=cols{n,1};
	for m=1:length(names)
		if strcmp(args.tokens{3},names{m}) col=cols{n,2}; break; end
	end
	if ~isempty(col) break; end
end

if isempty(col)
	warning(['Unrecognised colour "' args.tokens{3} '", black assumed'])
	out=[0 0 0];
	return;
end

switch args.tokens{1}
case 'slightly', fac=0.75;
case 'normal', fac=0.5;
case 'very', fac=0.25;
case 'extremely', fac=0.125;
otherwise
	warning(['Unrecognised modifier "' args.tokens{1} '", normal assumed'])
	fac=0.5;
end

switch args.tokens{2}
case {'light','pale'}, out=1-(1-col)*fac;
case 'normal', out=col;
case 'dark', out=col*fac;
otherwise
	warning(['Unrecognised descriptor "' args.tokens{2} '", normal assumed'])
	out=col;
end

function out=translatecolorRGB(in)

BGR=0;
if in<0
	in=-in;
	BGR=1;
end

b=bytes4(in);
if BGR out=b(4:-1:2); else out=b(2:4); end

function out=qbcolor(in)

% rgb value from basic colour code
% 0-7 are normal, 8-15 are bright
% 0 - black
% 1 - red,  2 - green,   3 - blue
% 4 - cyan, 5 - magenta, 6 - yellow
% 7 - white

bright=0.5;
if in>7 in=in-8; bright=1; end

switch in
case 0, rgb=[0 0 0];
case 1, rgb=[1 0 0];
case 2, rgb=[0 1 0];
case 3, rgb=[0 0 1];
case 4, rgb=[0 1 1];
case 5, rgb=[1 0 1];
case 6, rgb=[1 1 0];
case 7, rgb=[1 1 1];
otherwise
	warning('Unrecognised QBasic color, black assumed');
	out=[0 0 0];
	return;
end

out=rgb*bright;


function tokens=rgb_parse(str)

% parse string to obtain all tokens
% quoted strings count as single tokens

inquotes=0;
intoken=0;
pos=1;
l=length(str);
st=0;
ed=0;
token='';
tab=char(9);
tokens=cell(0);
while(pos<=l)
	ch=str(pos);
	if inquotes
		if ch=='"'
			inquotes=0;
			tokens={tokens{:} token};
		else
			token=[token ch];
		end
	elseif intoken
		if ch==' ' | ch==tab
			intoken=0;
			tokens={tokens{:} token};
		elseif ch=='"'
			error(['Quote misplace in <' str '>']);
		else
			token=[token ch];
		end
	else
		if ch==' ' | ch==tab
			% do nothing
		elseif ch=='"'
			token='';
			inquotes=1;
		else
			token=ch;
			intoken=1;
		end
	end
	pos=pos+1;
end

if intoken tokens={tokens{:} token}; end
if inquotes error(['Unpaired quotes in <' str '>']); end

tokens=tokens';



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEMO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rgb_demo

figure(1)
clf
cols = get_cols;
cols = {cols{:,1}}';
cols = { cols{:}, ...
	'k', ...
	'r', ...
	'g', ...
	'b', ...
	'y', ...
	'm', ...
	'c', ...
	'w', ...
	'', ...
	'extremely dark green', ...
	'very dark green', ...
	'dark green', ...
	'slightly dark green', ...
	'green', ...
	'slightly pale green', ...
	'pale green', ...
	'very pale green', ...
	'extremely pale green', ...
};

height=9;
x=0;
y=0;
for n=1:length(cols)
	rect(x,y,cols{n})
	y=y+1;
	if y==height
		x=x+2;
		y=0;
	end
end
if y==0 x=x-2; end
axis([0 (x+2) 0 height])
title('names on different rows are alternates')

function rect(x,y,col)
if isempty(col) return; end
r=rectangle('position',[x+0.1 y+0.1 1.8 0.8]);
col_=col;
if iscell(col) col=col{1}; end
colrgb=rgb(col);
if strcmp(col(1),'u') & length(col)==2
	t=text(x+1,y+0.5,{'unnamed',['colour (' col(2) ')']});
	set(r,'facecolor',colrgb);
else
	t=text(x+1,y+0.5,col_);
	set(r,'facecolor',colrgb);
	if sum(colrgb)<1.5 set(t,'color',[1 1 1]); end
end
set(t,'horizontalalignment','center')
set(t,'fontsize',10)

function rgb_list
cols=get_cols;
disp(' ')
for n=1:size(cols,1)
	code=cols{n,2};
	str=cols{n,1};
	str_=[];
	for m=1:length(str)
		str_=[str_ str{m} ', '];
	end
	str_=str_(1:end-2);
	if strcmp(str_(1),'u') & length(str_)==2
		str_=['* (' str_(2) ')'];
	end
	disp(['  [' sprintf('%.1f  %.1f  %.1f',code) '] - ' str_])
end
disp([10 '* colours marked thus are not named - if you know their' 10 '  designation, or if you feel sure a colour is mis-named,' 10 '  email me (address via help) or comment at' 10 '  www.mathworks.com/matlabcentral - "rgb demo" to see them' 10])

function cols=get_cols

cols={
	'black', [0 0 0]; ...
	'navy', [0 0 0.5]; ...
	'blue', [0 0 1]; ...
	'u1', [0 0.5 0]; ...
	{'teal','turquoise'}, [0 0.5 0.5]; ...
	'slateblue', [0 0.5 1]; ...
	{'green','lime'}, [0 1 0]; ...
	'springgreen', [0 1 0.5]; ...
	{'cyan','aqua'}, [0 1 1]; ...
	'maroon', [0.5 0 0]; ...
	'purple', [0.5 0 0.5]; ...
	'u2', [0.5 0 1]; ...
	'olive', [0.5 0.5 0]; ...
	{'gray','grey'}, [0.5 0.5 0.5]; ...
	'u3', [0.5 0.5 1]; ...
	{'mediumspringgreen','chartreuse'}, [0.5 1 0]; ...
	'u4', [0.5 1 0.5]; ...
	'sky', [0.5 1 1]; ...
	'red', [1 0 0]; ...
	'u5', [1 0 0.5]; ...
	{'magenta','fuchsia'}, [1 0 1]; ...
	'orange', [1 0.5 0]; ...
	'u6', [1 0.5 0.5]; ...
	'u7', [1 0.5 1]; ...
	'yellow', [1 1 0]; ...
	'u8', [1 1 0.5]; ...
	'white', [1 1 1]; ...
	};

for n=1:size(cols,1)
	if ~iscell(cols{n,1}) cols{n,1}={cols{n,1}}; end
end

% converts a DWORD into a four byte row vector
function out=bytes4(in)

out=[0 0 0 0];
if in<0 | in>(2^32-1)
	warning('DWORD out of range, zero assumed');
	return;
end

N=4;
while(in>0)
	out(N)=mod(in,256);
	in=(in-out(N))/256;
	N=N-1;
end
