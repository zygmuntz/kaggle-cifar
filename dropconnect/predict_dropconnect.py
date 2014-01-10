'produce a submission file from dropnn predictions'
'needs Python 2.7 for dictionary comprehension (labels_dict)'

import csv
import sys
import os
#import cPickle as pickle
import numpy as np

# cuda-conv way
from util import pickle, unpickle

output_file = sys.argv.pop()

num_args = len( sys.argv )
num_nets = num_args - 1
assert( num_nets > 0 )

#

label_names = [ 'airplane', 'automobile', 'bird',	'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]
labels_dict = { i: x for i, x in enumerate( label_names ) }

writer = csv.writer( open( output_file, 'wb' ))
writer.writerow( [ 'id', 'label' ] )
counter = 1

###

# 0th net
# result['labels']
# result['preds']
result = unpickle( sys.argv[1] ) 
num_batches = len( result['labels'] )

for ii in range( num_nets - 1 ):
	result_ii = unpickle( sys.argv[ii+2] )
	# check num of batches is consistant
	num_batches_ii = len( result_ii['labels'] )
	for jj in range( num_batches ):
		# check label is consistant
		assert( np.array_equal( result_ii['labels'][jj], result['labels'][jj] ))
		# nc result['pred'][jj]
		result['preds'][jj] += result_ii['preds'][jj]
		
#print len( result['preds'] )
#pickle( output_file, result )

for i in range( len( result['preds'] )):
	#print result['preds'][i].shape

	label_indexes = np.argmax( result['preds'][i], axis = 1 )
	#print label_indexes

	for i in label_indexes:
		label = labels_dict[i]
		writer.writerow( [ counter, label ] )
		counter += 1
		
assert( counter == 300001 )







