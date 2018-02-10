from syft.interfaces.keras import actual_keras

def load_data():
	return actual_keras.keras.datasets.fashion_mnist.load_data()