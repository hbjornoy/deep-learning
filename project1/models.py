# Pytorch neural net models

# IMPORTS
from torch import Tensor, nn



#Models

class Linear_regression_model(nn.Module):
	"""
	Simple Linear regression model with sigmoid activation
	"""
	def __init__(self, input_dim, output_dim):
		super(Linear_regression_model, self).__init__() # Inherited from the parent class nn.Module
		self.fc1 = nn.Linear(input_dim, output_dim) # 1st Full-Connected Layer: 784 (input data) -> 10 (output class
		self.sig = nn.Sigmoid()

	def forward(self, x): # Forward pass: stacking each layer together
		out = self.fc1(x)
		out = self.sig(out)
		return out


def DenseNet(input_dim, output_dim, nb_hidden_layers=1, hidden_width=100, dropout_rate=False):
	"""
	Fully connected feedforward neural net with adjustible, but uniform width of hidden layers and and adjustable depth
	"""
	if nb_hidden_layers < 1:
		print("you need at least one hidden layer")
		model = None
	else:
		layers = []
		layers.append(nn.Linear(input_dim, hidden_width))
		layers.append(nn.ReLU())
		for i in range(nb_hidden_layers-1):
			layers.append(nn.Linear(hidden_width, hidden_width))
			layers.append(nn.ReLU())
			if type(dropout_rate) == float and dropout_rate > 0.0 and dropout_rate < 1.0:
				layers.append(nn.Dropout(p=dropout_rate))

		layers.append(nn.Linear(hidden_width, output_dim))
		layers.append(nn.Sigmoid())

		model = nn.Sequential(*layers)
		return model
