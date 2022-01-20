from torch import nn, optim

def create_model():
    #TODO: Build a feed-forward network. You can use the network you built in previous exercises
    input_size = 784 #28x28
    output_size = 10
    model =nn.Sequential(nn.Linear(input_size,128), #Performs W.x + b
                          nn.ReLU(),               
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64,output_size),
                          nn.LogSoftmax(dim=1))

    return model

#NOTE: Do not change any of the variable names to ensure that the training script works properly

model=create_model()

cost = nn.NLLLoss() #TODO: Add your cost function here

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #TODO: Add your optimizer here

