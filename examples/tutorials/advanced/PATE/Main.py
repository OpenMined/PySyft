# Required imports
import torch
from Teacher import Teacher
from Model import Model
from data import load_data, NoisyDataset
from util import accuracy, split
from Student import Student
import syft as sy
from syft.frameworks.torch.differential_privacy import pate

class Arguments:

    # Class used to set hyperparameters for the whole PATE implementation
    def __init__(self):

        self.batchsize = 64
        self.test_batchsize = 10
        self.epochs = 50
        self.student_epochs = 15
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.n_teachers = 50
        self.save_model = False


args = Arguments()

train_loader = load_data(True, args.batchsize)
test_loader = load_data(False, args.test_batchsize)

# Declare and train teachers on MNIST training data
teacher = Teacher(args, Model, n_teachers=args.n_teachers)
teacher.train(train_loader)

# Evaluate Teacher accuracy
teacher_targets = []
predict = []

counts=[]
original_targets=[]


for data, target in test_loader:
    
    output=teacher.predict(data)
    
    arr_target=[]
    teacher_targets.append(target)
    original_targets.append(target)
    predict.append(output['predictions'])
    counts.append(output['model_counts'])

print("Accuracy: ", accuracy(torch.tensor(predict),teacher_targets))

print("\n")
print("\n")

print("Training Student")

print("\n")
print("\n")

# Split the test data further into training and validation data for student
train, val = split(test_loader, args.batchsize)

student = Student(args, Model())
N = NoisyDataset(train, teacher.predict)
student.train(N)

results = []
targets = []

total = 0.0
correct = 0.0

for data, target in val:

    predict_lol = student.predict(data)
    correct += float((predict_lol == (target)).sum().item())
    total += float(target.size(0))

print("Private Baseline: ", (correct / total) * 100)

counts_lol=torch.stack(counts).contiguous().view(50,10000)
predict_lol=torch.tensor(predict).view(10000)

data_dep_eps,data_ind_eps=teacher.analyze(counts_lol,predict_lol,moments=20)
print("Epsilon: ",teacher.analyze(counts_lol,predict_lol))