import hypertools as hyp
data = hyp.tools.load('mushrooms')
hypO = hyp.plot(data, reduce_model='IncrementalPCA')
