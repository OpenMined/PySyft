class GPU:

    def __init__(self, name:str, count:int, manufacturer:str, memory_total:int, memory_per_gpu:int):
        self.name = name
        self.count = count
        self.manufacturer = manufacturer
        self.memory_total = memory_total
        self.memory_per_gpu = memory_per_gpu