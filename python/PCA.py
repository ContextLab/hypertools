

import numpy as np 
from sklearn.decomposition import PCA

def do_pca(x, y):
	pca=PCA(n_components=3)
	pca.fit(x)
	#fit takes one array

	reduced=pca.transform(y)
	#tranforms takes multiple arrays


def align(*args):
	for x in range(0, len(args)):
		if x==0:
			template=args[x]
		else:
			next=do_pca(np.transpose(template/x),np.transpose(args[x]))
			template=template+np.transpose(next)

	template=template/len(args)


	#STEP 2: NEW COMMON TEMPLATE 
	template2= np.zeros(template.shape)
	for x in range(0, len(args)):
		next = do_pca(np.transpose(template),np.transpose(args[x]))
		template2 = template2 + np.transpose(next)

	template2=template2/len(args)


	empty= np.zeros(template2.shape)
	aligned=[empty]*(len(args)) 
	#STEP 3 (below): ALIGN TO NEW TEMPLATE
	for x in range(0, len(args)):
		next = do_pca(np.transpose(template2),np.transpose(args[x]))
		aligned[x] = np.transpose(next)

	return aligned
	print aligned