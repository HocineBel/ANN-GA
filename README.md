# ANN-GA
import tensorflow as tf
import os
import numpy
import genetic
import ann
import csv
import numpy as np
import random
import data_input
import gc
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt

NUM_PARALLEL_EXEC_UNITS = 4
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads = NUM_PARALLEL_EXEC_UNITS, 
         inter_op_parallelism_threads = 2, 
         allow_soft_placement = True, 
         device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS })
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
num_generations = 4
sol_per_pop = 4
num_parents_mating = 4
mutation_percent = 2
#ANN hyperparameter definition search space
layers_list = [1,2,3]
#layers_list = [2,3,4]
batch_list = [10, 25, 50, 100, 200]
optimisers = ['Adam', 'Adagrad', 'RMSprop', 'sgd']
kernel_initializer = ['uniform','normal']
epochs = [50, 100, 150, 200]
dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
training = [0.05, 0.10,0.15,0.20,0.25,0.30]
activation = ['relu', 'tanh', 'sigmoid', 'elu']
#Creating an empty list to store the initial population
initial_population = []
#Creating an empty list to store the final solutions
final_list=[]
#Create initial population
for curr_sol in numpy.arange(0, sol_per_pop):
    layers= random.choice(layers_list)
    batch = random.choice(batch_list)
    opt = random.choice(optimisers)
    ker = random.choice(kernel_initializer)
    epo = random.choice(epochs)
    drop = random.choice(dropout)
    train = random.choice(training)
    act = random.choice(activation)
    if layers == 1:  
        n1 = random.randint(1,10)
        neurons = n1
    elif layers == 2: 
        n1 = random.randint(1,10)
        n2 = random.randint(1,10)
        neurons = n1,n2
    elif layers == 3: 
        n1 = random.randint(1,10)
        n2 = random.randint(1,10)
        n3 = random.randint(1,10)
        neurons = n1,n2,n3
       
    initial_population.append([layers, neurons, batch, opt, ker, epo, drop, train, act])
#Initial population
pop_inputs = np.asarray(initial_population)
del(initial_population)
#Start GA process
for generation in range(num_generations):    
    pre_list=[]
    list_inputs =[]
    list_fitness=[]
    list_objective=[]
    list_other_metrics = []
    print("================================================================")
    print("================================================================")
    print("\nGeneration : ", generation+1)
    print("Inputs : \n",  pop_inputs)
    pop_inputs = pop_inputs                                 
    # Measuring the fitness of each solution in the population.
    fitness = []
    objective = []
    other_metrics =[]
    
    #ANN model training for sol_population p in generation g
    for index in range(sol_per_pop):
        print('\n Generation: ', generation+1, " of ", num_generations, ' Simulation: ', index+1 ,' of ', sol_per_pop)
        X_train, X_test, Y_train, Y_test = data_input.data(pop_inputs[index][7])
        print('\n Test/Training :', ((1-pop_inputs[index][7])*100),'/',pop_inputs[index][7]*100)
        #Export ANN metric performance
        RMSE, RMSE_val, mae, val_mae, R2, R2_v = ann.model_ANN(X_train.shape[1], Y_train.shape[1], pop_inputs[index][0], pop_inputs[index][1],\
                                   pop_inputs[index][2], pop_inputs[index][3], pop_inputs[index][4], pop_inputs[index][5], pop_inputs[index][6], pop_inputs[index][8],
                                   X_train, Y_train, X_test, Y_test)
        obj = ((1-RMSE)*.5 + (1-RMSE_val)*.5)
        fitness.append([RMSE, RMSE_val])
        objective.append([obj])
        print("Fitness")
        print(RMSE, RMSE_val)
        print("Objective")
        print(obj)
        other_metrics.append([mae, val_mae, R2, R2_v])
        del  X_train, Y_train, X_test, Y_test
        gc.collect()
    print(fitness)
    print(objective)
    list_fitness.append(fitness)
    list_objective.append(objective)
    list_inputs.append(pop_inputs.tolist())
    list_other_metrics.append(other_metrics)
    # top performance ANN model in the population are selected for mating.
    parents = genetic.mating_pool(pop_inputs, objective.copy(), num_parents_mating)
    print("Parents")
    print(parents)
    parents = numpy.asarray(parents) 
    # Crossover to generate the next geenration of solutions
    offspring_crossover = genetic.crossover(parents,offspring_size=(int(num_parents_mating/2), pop_inputs.shape[1]))
    print("Crossover")
    print(offspring_crossover)
    # Mutation for population variation
    offspring_mutation = genetic.mutation(offspring_crossover, sol_per_pop, num_parents_mating, mutation_percent=mutation_percent)
    print("Mutation")
    print(offspring_mutation) 
    # New population for generation g+1
    pop_inputs[0:len(offspring_crossover), :] = offspring_crossover
    pop_inputs[len(offspring_crossover):, :] = offspring_mutation
    print('NEW INPUTS :\n', pop_inputs )
       
    for x in range(len(list_inputs)):
        for y in range(len(list_inputs[0])):
            pre_list = list_inputs[x][y]
            for m in range(len(list_fitness[x][y])):
                pre_list.append(list_fitness[x][y][m])
            pre_list.append(list_objective[x][y][0])
            for w in range(len(list_other_metrics[x][y])):
                pre_list.append(list_other_metrics[x][y][w])
            final_list.append(pre_list)      
        del(fitness, objective, other_metrics, parents, offspring_mutation, offspring_crossover, list_inputs, list_fitness, list_objective, list_other_metrics, pre_list)
    gc.collect()
#Insert headers to final list
final_list.insert(0, ['Layers', 'Neurons', 'batch', 'optimiser', 'keras', 'epochs', 'dropout', 'train %', 'activation', 'RMSE', 'VAL_RMSE', 'Objective', 'mae', 'val_mae', 'R2', 'R2_v' ])
final_list
#Saving all ANN structures, hyperparameters and metrics
with open('FINAL_RESULTS.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerows(final_list)
#-------plot     
model.save_weights('D:\Full_Auto_Hoo_Py_V_1\Py_Trn\traitedData\test\weights.h5')
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
tf.keras.utils.plot_model(model, 'model.png', show_shapes=True)
model.summary()

from keras.models import load_model
# Save the model
model.save("my_model.h5")
# Save the weights
model.save_weights("my_model_weights.h5")

# Load the model
loaded_model = load_model("my_model.h5")
# Load the weights
loaded_model.load_weights("my_model_weights.h5")



import seaborn as sns

# Create a dataframe with the example data
data = {'Kernel/Weight Initializer': kernel_initializers, 'Performance': scores}
df = pd.DataFrame(data)

# Create a bar chart with a separate plot for each optimizer and layer configuration
sns.catplot(x='Kernel/Weight Initializer', y='Performance', data=df, kind='bar', col='Optimizer', row='Layer')
plt.show()

import matplotlib.pyplot as plt
# calculate MSE and RMSE
mse = mean_squared_error(y_true, y_pred)
rmse = sqrt(mse)
# create a plot
plt.plot(mse, label='MSE')
plt.plot(rmse, label='RMSE')
# add labels and legend
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
# show the plot
plt.show()


import matplotlib.pyplot as plt
# calculate fitness values for different batch sizes
batch_sizes = [32, 64, 128, 256]
fitness_values = []
for batch_size in batch_sizes:
    model.batch_size = batch_size
    model.fit(X, y)
    fitness_values.append(model.evaluate(X_val, y_val))
# create a plot
plt.plot(batch_sizes, fitness_values)
# add labels and legend
plt.xlabel('Batch Size')
plt.ylabel('Fitness')
# show the plot
plt.show()


import matplotlib.pyplot as plt
# calculate fitness values for different kernel/weight initialisers
kernel_initialisers = ['uniform', 'normal', 'glorot_normal']
fitness_values = []
for kernel_initialiser in kernel_initialisers:
    model.kernel_initializer = kernel_initialiser
    model.fit(X, y)
    fitness_values.append(model.evaluate(X_val, y_val))
# create a plot
plt.plot(kernel_initialisers, fitness_values)
# add labels and legend
plt.xlabel('Kernel/Weight Initialiser')
plt.ylabel('Fitness')


import matplotlib.pyplot as plt
# predict values using the optimal ANN model
y_pred = model.predict(X_val)
# calculate residuals
residuals = y_pred - y_val
# create a scatter plot of the predicted vs actual values for Y1
plt.scatter(y_val[:,0], y_pred[:,0])
# add labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Optimal ANN Model Prediction Performance for Y1')
# show the plot
plt.show()
# create a line plot of the residuals for Y1
plt.plot(residuals[:,0])
# add labels and title
plt.xlabel('Samples')
plt.ylabel('Residuals')
plt.title('Optimal ANN Model Residuals for Y1')
# show the plot
plt.show()



#data #--------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gc
def data(training):
    dataframe = pd.read_csv("Data.csv", sep=',', header = None)
    dataset = dataframe.values  
    X = dataset[:,:-2]  #  all columns except for the last two.
    Y = dataset[:,-2:]  #  the output variables are the last two columns in the dataset.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=training, random_state=42)
    del(dataframe, X, Y) #removing them from memory can help to free up space and improve the performance of your program.
    gc.collect()  
    scaler = StandardScaler()   # created scaler
    scaler.fit(Y_train)     # fit scaler on training dataset
    Y_train = scaler.transform(Y_train)   # transform training dataset
    Y_test = scaler.transform(Y_test)    # transform test dataset
    return X_train, X_test, Y_train, Y_test
#GENETIC ALGORITHM  #--------------------------------------------------------------
import numpy
import random  #GENETIC ALGORITHM OPERATORS
def mating_pool(pop_inputs, objective, num_parents):  #Mating function   
    objective = numpy.asarray(objective)
    parents = [[None,None,None, None, None, None, None, None, None]]* num_parents
    for parent_num in range(num_parents):
        best_fit_index = numpy.where(objective == numpy.max(objective))
        best_fit_index = best_fit_index[0][0]
        parents[parent_num] = pop_inputs[best_fit_index, :]
        objective[best_fit_index] = -9999999
    return parents
def crossover(parents, offspring_size):   #Crossover function
    offspring = [[None,None,None, None, None, None, None, None, None]]* offspring_size[0]
    crossover_loc = numpy.uint32(offspring_size[1]/2)
    parents_list = parents.tolist()
    for k in range(offspring_size[0]):       # Loc first parent
        parent_1_index = k%parents.shape[0]  # Loc second parent
        parent_2_index = (k+1)%parents.shape[0]      # Offspring generation
        offspring[k] = parents_list[parent_1_index][0:crossover_loc] + parents_list[parent_2_index][crossover_loc:]
    return offspring
def mutation(offspring_crossover, sol_per_pop, num_parents_mating, mutation_percent):
    offspring_crossover_a = numpy.asarray(offspring_crossover) # convert to array to do shape calculations
    num_mutations = numpy.uint32((mutation_percent*offspring_crossover_a.shape[1])/100)
    mutation_indices = numpy.array(random.sample(range(0, offspring_crossover_a.shape[1]), num_mutations))
    offspring_mutation = offspring_crossover * sol_per_pop
    offspring_mutation = offspring_mutation [:sol_per_pop-offspring_crossover_a.shape[0]]
    offspring_mutation = numpy.asarray(offspring_mutation, dtype=object)
    for index in range(sol_per_pop-int(num_parents_mating/2)):
        if 0 in mutation_indices: 
            if 1 not in mutation_indices:
                value = random.randint(1,3)
                offspring_mutation[index, 0] = value
                if value == 1:      # n1 = random.randint(10,10)
                   n1 = random.randint(1,10)    # n = n1
                elif value == 2: 
                    n1 = random.randint(1,10)
                    n2 = random.randint(1,10)
                    n = n1,n2
                elif value == 3: 
                    n1 = random.randint(1,10)
                    n2 = random.randint(1,10)
                    n3 = random.randint(1,10)
                    n = n1,n2,n3
                
        elif [0 and 1] in mutation_indices:
            value = random.randint(1,3)
            offspring_mutation[index, 0] = value
            if value == 1:  
                n1 = random.randint(1,10)
                n = n1
            elif value == 2: 
                n1 = random.randint(1,10)
                n2 = random.randint(1,10)
                n = n1,n2
            elif value == 3: 
                n1 = random.randint(1,10)
                n2 = random.randint(1,10)
                n3 = random.randint(1,10)
                n = n1,n2,n3
            offspring_mutation[index, 1] = n
        
        if 1 in mutation_indices:
            if 0 not in mutation_indices:
                value = random.randint(1,3)
                offspring_mutation[index, 0] = value
                
                if value == 1:  
                    n1 = random.randint(1,10)
                    n = n1
                elif value == 2: 
                    n1 = random.randint(1,10)
                    n2 = random.randint(1,10)
                    n = n1,n2
                elif value == 3: 
                    n1 = random.randint(1,10)
                    n2 = random.randint(1,10)
                    n3 = random.randint(1,10)
                    n = n1,n2,n3
                             
                offspring_mutation[index, 1] = n
            
        if 2 in mutation_indices:
            b = [10, 25, 50, 100, 100]
            value = random.choice(b)
            offspring_mutation[index, 2] = value
            
        if 3 in mutation_indices:
            o = ['Adam', 'Adagrad', 'RMSprop', 'sgd']
            value = random.choice(o)
            offspring_mutation[index, 3] = value

        if 4 in mutation_indices:
            k = ['uniform','normal']
            value = random.choice(k)
            offspring_mutation[index, 4] = value
            
        if 5 in mutation_indices:
            e = [50, 100, 150, 200]
            value = random.choice(e)
            offspring_mutation[index, 5] = value

        if 6 in mutation_indices:
            d = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
            value = random.choice(d)
            offspring_mutation[index, 6] = value
            
        if 7 in mutation_indices:
            t = [0.05, 0.10,0.15,0.20,0.25,0.30]
            value = random.choice(t)
            offspring_mutation[index, 7] = value
        
        if 8 in mutation_indices:
            at = ['relu', 'tanh', 'sigmoid', 'elu']
            value = random.choice(at)
            offspring_mutation[index, 8] = value
     
    return offspring_mutation
#ANN model #--------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
from keras.layers import Dropout
import math
#Defining ANN model
def model_ANN(inputs, outputs, layers, neurons, batch, opt, ker, epo, drop, act, x_t, y_t, x_v, y_v):
    model = Sequential()
    if layers == 1:  
        model.add(Dense(neurons, input_dim= inputs, kernel_initializer=ker, activation=act))
        model.add(Dropout(drop))
        model.add(Dense(outputs, kernel_initializer=ker,activation='linear'))
        n = (neurons,)
         
    elif layers ==2:    
        model.add(Dense(neurons[0], input_dim= inputs, kernel_initializer=ker, activation=act))
        model.add(Dropout(drop))
        model.add(Dense(neurons[1], kernel_initializer=ker, activation=act))
        model.add(Dropout(drop))
        model.add(Dense(outputs, kernel_initializer=ker,activation='linear'))
        n = (neurons[0], neurons[1])
            
    elif layers ==3:    
        model.add(Dense(neurons[0], input_dim= inputs, kernel_initializer=ker, activation=act))
        model.add(Dropout(drop))
        model.add(Dense(neurons[1], kernel_initializer=ker, activation=act))
        model.add(Dropout(drop))
        model.add(Dense(neurons[2], kernel_initializer=ker, activation=act))
        model.add(Dropout(drop))
        model.add(Dense(outputs, kernel_initializer=ker,activation='linear'))
        n = (neurons[0], neurons[1], neurons[2])

    print('No Inputs:', inputs, '  No Outputs: ', outputs)
    print('Layers: ', layers, '  Neurons: ', n )
    print('Batch: ', batch, '  Optimizer: ', opt , '  Initializer: ' , ker )
    print('Epochs: ', epo, '  Dropout: ', drop, ' Activation: ', act )
    model.compile(loss='mean_squared_error', optimizer= opt, metrics=['mae'])
    history = model.fit(x_t, y_t, epochs= epo, batch_size= batch, verbose=1, validation_data=(x_v, y_v))
    predictions = model.predict(x_t)
    predictions_v =  model.predict(x_v)
#ANN Performance metrics list
    MSE_scaled = mean_squared_error(y_t, predictions)
    RMSE = math.sqrt(MSE_scaled)
    MSE_scaled_val = mean_squared_error(y_v, predictions_v)
    RMSE_val = math.sqrt(MSE_scaled_val)
    R2 = r2_score(y_t, predictions)
    R2_v = r2_score(y_v, predictions_v)
    mae = history.history['mae'][-1]
    val_mae = history.history['val_mae'][-1]
    print("Results-- RMSE: %.2f RMSEv: %.2f R2: %.2f R2v: %.2f MAE: %.2f " % (RMSE, RMSE_val, R2, R2_v, mae))
    return RMSE, RMSE_val, mae, val_mae, R2, R2_v
