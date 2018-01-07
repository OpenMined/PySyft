from syft.interfaces.keras import actual_keras

def to_categorical(*args):
	return actual_keras.keras.utils.to_categorical(*args)