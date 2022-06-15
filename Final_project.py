#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
import pandas as pd
import numpy as np
import random
import math

#SelectionType
Deterministic = 1
Stochastic= 2

#計算總共的時間
def compute_objective_value(jobs,job_machine):
    total_time = 0
    for i,job in enumerate(jobs):
        total_time += job_machine[job][i+1]
    return total_time

#初始化個參數
def initialize(pop_size,number_of_genes,total_size):
    global selected_chromosomes
    global indexs
    global chromosomes
    global fitness
    global objective_values
    global best_chromosome
    global best_fitness
    selected_chromosomes = np.zeros((pop_size,number_of_genes))
    indexs = np.arange(total_size)
    chromosomes = np.zeros((total_size,number_of_genes),dtype=int)
    for i in range(pop_size):
        for j in range(number_of_genes):  
            chromosomes[i][j] = j
        np.random.shuffle(chromosomes[i])
       
    for i in range(pop_size,total_size):
        for j in range(number_of_genes):
            chromosomes[i][j] = -1
                
    fitness = np.zeros(total_size) 
    objective_values = np.zeros(total_size)
    best_chromosome = np.zeros(number_of_genes,dtype=int)
    best_fitness = 0

#計算適應度
def evaluate_fitness(pop_size,job_machine,least_fitness_factor):
    global chromosomes
    global objective_values
    global fitness
    for i,chromosome in enumerate(chromosomes[:pop_size]):
        objective_values[i] = compute_objective_value(chromosome,job_machine)
           
    min_obj_val = np.min(objective_values)
    max_obj_val = np.max(objective_values)
    range_obj_val = max_obj_val-min_obj_val
        
    for i,obj in enumerate(objective_values):
        fitness[i] = max(least_fitness_factor*range_obj_val,pow(10,-5))+(max_obj_val-obj)

#更新適應度最好的染色體
def update_best_solution():
    global fitness
    global best_fitness
    global chromosomes
    global best_chromosome
    best_index = np.argmax(fitness)
    if(best_fitness<fitness[best_index]):
        best_fitness = fitness[best_index]
        for i,gene in enumerate(chromosomes[best_index]):
            best_chromosome[i] = gene

#打亂index的順序
def shuffle_index(length):
    global indexs
    for i in range(length):
        indexs[i] = i
    np.random.shuffle(indexs[:length])
    
#Partially matched crossover
def perform_crossover_operation(pop_size,crossover_size,job_machine,number_of_genes):
    global indexs
    global objective_values
    global chromosomes
    shuffle_index(pop_size)
        
    child1_index = pop_size
    child2_index = pop_size+1
    count_of_crossover = int(crossover_size/2)
    for i in range(count_of_crossover):
        parent1_index = indexs[i]
        parent2_index  = indexs[i+1]
            
        #if(crossover_type == PartialMappedCrossover):
        partial_mapped_crossover(parent1_index,parent2_index,child1_index,child2_index,number_of_genes)
        objective_values[child1_index] = compute_objective_value(chromosomes[child1_index],job_machine)
        objective_values[child2_index] = compute_objective_value(chromosomes[child2_index],job_machine)
            
        child1_index +=2
        child2_index +=2
        
def partial_mapped_crossover(p1,p2,c1,c2,number_of_genes):
    global mapping
    global chromosomes
    #reset
    for i in range(number_of_genes):
        mapping[i] = -1
         
    rand1 = random.randint(0,number_of_genes-2)
    rand2 = random.randint(rand1+1,number_of_genes-1)
       
    for i in range(rand1,rand2+1):
        c1_gene = chromosomes[p2][i] 
        c2_gene = chromosomes[p1][i]
            
        if(c1_gene==c2_gene):
            continue
            
        elif(mapping[c1_gene]==-1 and mapping[c2_gene]==-1):
            mapping[c1_gene] = c2_gene
            mapping[c2_gene] = c1_gene
                
        elif(mapping[c1_gene]==-1):
            mapping[c1_gene] =  mapping[c2_gene]
            mapping[mapping[c2_gene]] = c1_gene
            mapping[c2_gene] = -2
                
        elif (mapping[c2_gene]==-1):
            mapping[c2_gene] =  mapping[c1_gene]
            mapping[mapping[c1_gene]] = c2_gene
            mapping[c1_gene] = -2
                
        else:
            mapping[mapping[c1_gene]] = mapping[c2_gene]
            mapping[mapping[c2_gene]] = mapping[c1_gene]
            mapping[c1_gene] = -3
            mapping[c2_gene] = -3
                
    for i in range(number_of_genes):
        if(i>=rand1 and i<=rand2):
            chromosomes[c1][i] =  chromosomes[p2][i]
            chromosomes[c2][i] =  chromosomes[p1][i]
        else:
            if(mapping[chromosomes[p1][i]] >=0):
                chromosomes[c1][i] = mapping[chromosomes[p1][i]]
            else:
                chromosomes[c1][i] =chromosomes[p1][i]        
                
            if(mapping[chromosomes[p2][i]] >=0):
                chromosomes[c2][i] = mapping[chromosomes[p2][i]]
            else:
                chromosomes[c2][i] =chromosomes[p2][i]
                
#Selection(輪盤法)
def do_roulette_wheel_selection(fitness_list):
    global fitness
    sum_fitness = sum(fitness_list)
    transition_probability = [fitness/sum_fitness for fitness in fitness_list]
        
    rand = random.random()
    sum_prob = 0
    for i,prob in enumerate(transition_probability):
        sum_prob += prob
        if(sum_prob>=rand):
            return i

#Selection
def perform_selection(selection_type,pop_size,total_size,number_of_genes):
    global fitness
    global selected_chromosomes
    global chromosomes
    #確定型的選法
    if selection_type == Deterministic:
        index = np.argsort(fitness)[::-1]
    #輪盤法
    elif selection_type == Stochastic:
        index = [do_roulette_wheel_selection(fitness) for i in range(pop_size)]
        
    else:
        index = shuffle_index(total_size)
        
    for i in range(pop_size):
        for j in range(number_of_genes):
            selected_chromosomes[i][j] =  chromosomes[index[i]][j]
        
    for i in range(pop_size):
        for j in range(number_of_genes):
            chromosomes[i][j] = selected_chromosomes[i][j]

#Inversion mutation
def perform_mutation_operation(pop_size,crossover_size,mutation_size,job_machine,number_of_genes):
    global indexs
    global objective_values
    global chromosomes
    shuffle_index(pop_size+crossover_size)
    child1_index = pop_size+crossover_size
    for i in range(mutation_size):
        parent1_index = indexs[i]
        inversion_mutation(parent1_index,child1_index,number_of_genes)
        objective_values[child1_index] = compute_objective_value(chromosomes[child1_index],job_machine)
        child1_index += 1
def inversion_mutation(p1,c1,number_of_genes):
    global chromosomes
    rand1 = random.randint(0,number_of_genes-2)
    rand2 = random.randint(rand1+1,number_of_genes-1)
    for i in range(number_of_genes):
        if(i<rand1 or i>rand2):
            chromosomes[c1][i] = chromosomes[p1][i]
        else:
            index = rand2-(i-rand1)
            chromosomes[c1][i] = chromosomes[p1][index]

#按下button執行GA
def run_GA():
    global mapping
    label6=tk.Label(GA,text="                                                                                                      ")
    label6.grid(row=6,column=0,columnspan=2,padx=75,pady=20,sticky='w')
    label7=tk.Label(GA,text="                                                                              ")
    label7.grid(row=7,column=0,columnspan=2,padx=0,pady=0)
    label8=tk.Label(GA,text="                                                                               ")
    label8.grid(row=8,column=0,columnspan=2,padx=0,pady=0)
    label9=tk.Label(GA,text="                                                                                ")
    label9.grid(row=9,column=0,columnspan=2,padx=0,pady=0)
    label10=tk.Label(GA,text="                                                                                ")
    label10.grid(row=10,column=0,columnspan=2,padx=0,pady=0)
    label11=tk.Label(GA,text="                                                                               ")
    label11.grid(row=11,column=0,columnspan=2,padx=0,pady=0)
    label12=tk.Label(GA,text="                                                                                 ")
    label12.grid(row=12,column=0,columnspan=2,padx=0,pady=0)
    label13=tk.Label(GA,text="                                                                                 ")
    label13.grid(row=13,column=0,columnspan=2,padx=0,pady=0)
    label14=tk.Label(GA,text="                                                                                ")
    label14.grid(row=14,column=0,columnspan=2,padx=0,pady=0)
    label15=tk.Label(GA,text="                                                                                 ")
    label15.grid(row=15,column=0,columnspan=2,padx=0,pady=0)
    label16=tk.Label(GA,text="                                                                                 ")
    label16.grid(row=16,column=0,columnspan=2,padx=0,pady=0)
    label17=tk.Label(GA,text="                                                                                           ")
    label17.grid(row=17,column=0,columnspan=2,padx=0,pady=0)
    
    
    pop_size_tuple=('1.0','2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0','0.0')
    selection_type_tuple=('Deterministic','Stochastic')
    crossover_rate_tuple=('1','2','3','4','5','6','7','8','9')
    mutation_rate_tuple=('1','2','3','4','5','6','7','8','9')
    
    job_machine=data.values
    number_of_jobs=len(job_machine)
    compute_objective_value(range(len(data)),job_machine)
    
    pop_size_float = float(entry1.get())
    check_pop_size_float_tens_digit=str((pop_size_float//10)) in pop_size_tuple
    check_pop_size_float_units_digit=str((pop_size_float%10)) in pop_size_tuple
    if(check_pop_size_float_tens_digit==False or check_pop_size_float_units_digit==False):
        label6.config(text="Population size輸入錯誤，請輸入區間[1,99]的正整數               ",font=('Arial',10,'bold'))
    else:
        pop_size=int(float(entry1.get()))

        selection_type_str = entry2.get()
        check_selection_type_str = selection_type_str in selection_type_tuple
        if(check_selection_type_str==False):
            label6.config(text="Selection type輸入錯誤，請輸入Deterministic或Stochastic                ",font=('Arial',10,'bold'))
        else:
            if(selection_type_str=='Deterministic'):
                selection_type=1
            else:
                selection_type=2
            
            crossover_rate_str=entry3.get()
            crossover_rate_str_array = crossover_rate_str.split('.')
            if(len(crossover_rate_str_array)==1):
                crossover_rate_str_array.append('0')
            crossover_rate_int2 = int(crossover_rate_str_array[1])
            check_crossover_rate_int2=str(crossover_rate_int2) in crossover_rate_tuple
            if(check_crossover_rate_int2==False or crossover_rate_str_array[0]!='0'):
                label6.config(text="Crossover rate輸入錯誤，請輸入0.X，X=1~9                         ",font=('Arial',10,'bold'))
            else:
                crossover_rate=float(crossover_rate_str)
            
                mutation_rate_str=entry4.get()
                mutation_rate_str_array = mutation_rate_str.split('.')
                if(len(mutation_rate_str_array)==1):
                    mutation_rate_str_array.append('0')
                mutation_rate_int2 = int(mutation_rate_str_array[1])
                check_mutation_rate_int2=str(mutation_rate_int2) in mutation_rate_tuple
                if(check_mutation_rate_int2==False or mutation_rate_str_array[0]!='0'):
                    label6.config(text="Mutation rate輸入錯誤，請輸入0.X，X=1~9                         ",font=('Arial',10,'bold'))
                else:
                    mutation_rate=float(mutation_rate_str)
                
                crossover_size = int(pop_size*crossover_rate)
                if(crossover_size%2==1):
                    crossover_size -= 1
                mutation_size =  int(pop_size*mutation_rate)
                total_size = pop_size+mutation_size+crossover_size
                least_fitness_factor = 0.3
                mapping = [-1 for i in range(number_of_jobs)]

                for j in range(1):
                    print(f"round{j+1}:")
                    initialize(pop_size,number_of_jobs,total_size)

                    for i in range(100):
                        perform_crossover_operation(pop_size,crossover_size,job_machine,number_of_jobs)
                        perform_mutation_operation(pop_size,crossover_size,mutation_size,job_machine,number_of_jobs)
                        evaluate_fitness(pop_size,job_machine,least_fitness_factor)
                        update_best_solution()
                        perform_selection(selection_type,pop_size,total_size,number_of_jobs)
                        if(i %10 ==0):
                            print(F"iteration {i} :")
                            print(f"{best_chromosome}: {compute_objective_value(best_chromosome,job_machine)}")

                        if(i==99):
                            label6.config(text="本次執行結果:                                                                      ",font=('Arial',10,'bold'))
                            label7.config(text="第{}個工作指派給第0個人(工作時數{})".format(best_chromosome[0],job_machine[best_chromosome[0]][1]),font=('Arial',10))
                            label8.config(text="第{}個工作指派給第1個人(工作時數{})".format(best_chromosome[1],job_machine[best_chromosome[1]][2]),font=('Arial',10))
                            label9.config(text="第{}個工作指派給第2個人(工作時數{})".format(best_chromosome[2],job_machine[best_chromosome[2]][3]),font=('Arial',10))
                            label10.config(text="第{}個工作指派給第3個人(工作時數{})".format(best_chromosome[3],job_machine[best_chromosome[3]][4]),font=('Arial',10))
                            label11.config(text="第{}個工作指派給第4個人(工作時數{})".format(best_chromosome[4],job_machine[best_chromosome[4]][5]),font=('Arial',10))
                            label12.config(text="第{}個工作指派給第5個人(工作時數{})".format(best_chromosome[5],job_machine[best_chromosome[5]][6]),font=('Arial',10))
                            label13.config(text="第{}個工作指派給第6個人(工作時數{})".format(best_chromosome[6],job_machine[best_chromosome[6]][7]),font=('Arial',10))
                            label14.config(text="第{}個工作指派給第7個人(工作時數{})".format(best_chromosome[7],job_machine[best_chromosome[7]][8]),font=('Arial',10))
                            label15.config(text="第{}個工作指派給第8個人(工作時數{})".format(best_chromosome[8],job_machine[best_chromosome[8]][9]),font=('Arial',10))
                            label16.config(text="第{}個工作指派給第9個人(工作時數{})".format(best_chromosome[9],job_machine[best_chromosome[9]][10]),font=('Arial',10))
                            label17.config(text="本次GA最佳工作分配總時間為{}".format(math.floor(compute_objective_value(best_chromosome,job_machine)*10)/10.0),font=('Arial',10))

data = pd.read_csv("Jobs.csv")   
GA=tk.Tk()
GA.title('基因演算法求工作分配最佳解')
GA.geometry('550x550')

#Label
label1=tk.Label(GA,text="輸入各GA參數",font=('Arial',10,'bold'))
label1.grid(row=0,column=0,columnspan=2,padx=215,pady=20)
label2=tk.Label(GA,text='Population size :',font=('Arial',10))
label2.grid(row=1,column=0,padx=0,pady=0,sticky='e')
label3=tk.Label(GA,text='Selection type :',font=('Arial',10))
label3.grid(row=2,column=0,padx=0,pady=0,sticky='e')
label4=tk.Label(GA,text='Crossover rate :',font=('Arial',10))
label4.grid(row=3,column=0,padx=0,pady=0,sticky='e')
label5=tk.Label(GA,text='Mutation rate :',font=('Arial',10))
label5.grid(row=4,column=0,padx=0,pady=0,sticky='e')

#Entry
entry1=tk.Entry(GA,width=30)
entry1.grid(row=1,column=1,padx=0,pady=1,sticky='w')
entry2=tk.Entry(GA,width=30)
entry2.grid(row=2,column=1,padx=0,pady=1,sticky='w')
entry3=tk.Entry(GA,width=30)
entry3.grid(row=3,column=1,padx=0,pady=1,sticky='w')
entry4=tk.Entry(GA,width=30)
entry4.grid(row=4,column=1,padx=0,pady=1,sticky='w')

#Button
button=tk.Button(GA,text='開始執行',width=50,height=3,command=run_GA)
button.grid(row=5,column=0,columnspan=2,padx=5,pady=0)


GA.mainloop()

