# Python file with helpful functions for pytorch neural network coding

#IMPORTS
import dlc_bci as bci





#FUNCTIONS

def import_data(flatten=False):
    train_input , train_target = bci.load(root = './ data_bci')
    print ( str ( type ( train_input ) ) , train_input.size() )
    print ( str ( type ( train_target ) ) , train_target.size() )
    test_input , test_target = bci.load(root = './ data_bci', train = False)
    print ( str ( type ( test_input ) ) , test_input.size())
    print ( str ( type ( test_target ) ) , test_target.size())

    if flatten == True:
    	train_input = flatten_input_data(train_input)
    	test_input = flatten_input_data(test_input)
    
    return train_input, train_target, test_input, test_target


def flatten_input_data(input_data):
	"""
	flatten the dimensions with 28 features with 50 samples per feature into 1400 "features"
	"""
	return input_data.view(-1, input_data.size()[1]*input_data.size()[2])



