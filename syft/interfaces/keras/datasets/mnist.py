from syft.interfaces.keras import actual_keras

def load_data():
	return actual_keras.keras.datasets.mnist.load_data()