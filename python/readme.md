<h1>Hypertools - A python package for visualizing multidimensional data</h1>

![Hypertools example](images/hypertools.gif)

To install, navigate to this folder in Terminal and type:

pip install -e .

(this assumes you have pip installed on your system: https://pip.pypa.io/en/stable/installing/)

<h2>Main functions</h2>

+ <b>plot</b> - plots multidimensional data as static image or movie
+ <b>align</b> - align multidimensional data (See here for details)
+ <b>reduce</b> - implements PCA to reduce dimensionality of data
+ <b>describe</b> - plots/analyses to evaluate how well the functions above are working

<h2>Plot</h2>

<b>Inputs:</b>

A numpy array, or list of arrays

<b>Arguments:</b>

Format strings can be passed as a string, or tuple/list of length x.
See matplotlib API for more styling options

<b>Keyword arguments:</b>

<i>palette</i> (string): A matplotlib or seaborn color palette

<i>color</i> (list): A list of colors for each line to be plotted. Can be named colors, RGB values (e.g. (.3, .4, .1)) or hex codes. If defined, overrides palette. See http://matplotlib.org/examples/color/named_colors.html for list of named colors. Note: must be the same length as X.

<i>point_colors</i> (list of str, floats or ints): A list of colors for each point. Must be dimensionality of data (X). If the data type is numerical, the values will be mapped to rgb values in the specified palette.  If the data type is strings, the points will be labeled categorically.

<i>linestyle</i> (list): a list of line styles

<i>marker</i> (list): a list of marker types

See matplotlib API for more styling options

<i>labels</i> (list): A list of labels for each point. Must be dimensionality of data (X). If no label is wanted for a particular point, input `None`

<i>explore</i> (bool): Displays user defined labels or PCA coordinates on hover. When a point is clicked, the label will remain on the plot (dataarning: experimental feature, use at your own discretion!). To use, set `explore=True`.

<h3>Example uses</h3>

Import the library: `import hypertools as hyp`

Plot with default color palette: `hyp.plot(data)`

Plot as movie: `hyp.plot(data, animate=True)`

Change color palette: `hyp.plot(data,palette='Reds')`

Specify colors using unlabeled list of format strings: `hyp.plot([data[0],data[1]],['r:','b--'])`

Plot data as points: `hyp.plot([data[0],data[1]],'o')`

Specify colors using keyword list of colors (color codes, rgb values, hex codes or a mix): `hyp.plot([data[0],data[1],[data[2]],color=['r', (.5,.2,.9), '#101010'])`

Specify linestyles using keyword list: `hyp.plot([data[0],data[1],[data[2]],linestyle=[':','--','-'])`

Specify markers using keyword list: `hyp.plot([data[0],data[1],[data[2]],marker=['o','*','^'])`

Specify markers with format string and colors with keyword argument: `hyp.plot([data[0],data[1],[data[2]], 'o', color=['r','g','b'])`

Specify labels:
```
# Label first point of each list
labels=[]
for idx,i in enumerate(data):
    tmp=[]
    for iidx,ii in enumerate(i):
        if iidx==0:
            tmp.append('Point ' + str(idx))
        else:
            tmp.append(None)
    labels.append(tmp)

hyp.plot(data, 'o', labels=labels)
```

Specify point_colors:
```
# Label first point of each list
point_colors=[]
for idx,i in enumerate(data):
    tmp=[]
    for iidx,ii in enumerate(i):
            tmp.append(np.random.rand())
    point_colors.append(tmp)

hyp.plot(data, 'o', point_colors=point_colors)
```

Turn on explore mode (experimental): `hyp.plot(data, 'o', explore=True)`

<h2>Align</h2>

<b>Inputs:</b>

A list of numpy arrays

<b>Outputs</b>

An aligned list of numpy arrays

<h3>Example uses</h3>

align a list of arrays: `aligned_data = hyp.align(data)`

<h2>Reduce</h2>

<b>Inputs:</b>

A numpy array or list of numpy arrays

<b>Keyword arguments:</b>

ndims - dimensionality of output data

<b>Outputs</b>

An aligned list of numpy arrays

<h3>Example uses</h3>

Reduce n-dimensional array to 3d: `aligned_data = hyp.align(data, ndims=3)`

<h2>Describe</h2>

<b>Inputs:</b>

A numpy array or list of numpy arrays

<b>Outputs</b>

A plot summarizing the correlation between raw input data and PCA reduced data

<h3>Example uses</h3>

`hyp.describe(data)`

![Describe Example](images/describe_example.png)
