'''
Optimisation non différentiable et méthodes proximales -- TP1
Pirashanth RATNAMOGAN & Othmane SAYEM

Ce script continent notre implémentation pour le premier TP 
du cours d'Optimisation non différentiable et méthodes proximales 
de l'ENSTA ParisTech

Il contient l'implémentation d'un problème type LASSO résolu
par gradient proximal.
'''




import numpy as np
import pandas as pd

def generate_parameters(n,m,min_value_A,max_value_A):
	'''
	Fonction qui permet de générer des paramètres d'un problème de regression aléatoire
	Param: @n (int) correspond au nombre d'exempels du problème
	@m (int) correspond au nombre de features: les caractères descriptives du problème
	@min_value_A,max_value_A, (float,float) A et b prendront leurs valeurs entre min_value_A et max_value_A
	Return: les données A,b ainsi que le max des valeurs propres de A.T,A qui servira de pas de l'algorithme (voir rapport)
	'''
	A = np.random.rand(n,m)
	A = (max_value_A - min_value_A)*A - min_value_A
	b = (max_value_A - min_value_A)*np.random.rand(n,1)- min_value_A
	w,v = np.linalg.eigh(np.dot(A.T,A))
	L = np.max(w)
	return A,b,L

def generate_a_problem_with_u(n,m,min_value_A,max_value_A,nb_of_non_zero_coeff_u,noise_for_b):
	'''
	Fonction qui permet de générer des paramètres d'un problème de regression aléatoire et de donner le u optimal que l'on doit trouver
	Param: @n (int) correspond au nombre d'exempels du problème
	@m (int) correspond au nombre de features: les caractères descriptives du problème
	@min_value_A,max_value_A, (float,float) A et b prendront leurs valeurs entre min_value_A et max_value_A
	Return: les données A,b ainsi que le max des valeurs propres de A.T,A qui servira de pas de l'algorithme (voir rapport), ainsi que le u optimal
	a retrouver
	'''
	A = np.random.rand(n,m)
	A = (max_value_A - min_value_A)*A - min_value_A
	u = np.zeros((m,1)) 
	u[:nb_of_non_zero_coeff_u] =1
	np.random.shuffle(u)
	b = A.dot(u) + noise_for_b**np.random.rand(n,1)
	w,v = np.linalg.eigh(np.dot(A.T,A))
	L = np.max(w)
	return A,b,u,L
    


def thresolding_operator(u,epsilon,lambd):
	'''
	Operateur proximal de la norme 1
	Param:u: (float) coeffient à seuiller
	epsilon: (float) pas de l'algorithme
	lambd: (float) paramètre de régularisation de l'algorithme
	Return: le coeffient une fois seuillé
	'''
	if (u-epsilon*lambd)>0:
		return u - epsilon*lambd
	elif (u+epsilon*lambd)<0:
		return u + epsilon*lambd
	else:
		return 0

thresolding_operator_vector = np.vectorize(thresolding_operator) #permet d'applique l'opérateur de seuillage sur un vecteur directement
	

class GradientProximal:
	'''
	Class qui permet de faire tourner l'algorithme du gradient proximal afin de résoudre le problème du Lasso.
	C'est à dire minimiser $\min_u ||Au -b||_2^2 + \lambda |u|_1 $.
 	'''
	
	def __init__(self,epsilon,A,b,lambd,crit_arret=1e-4,criterion ='grad'):
		'''
		Constructeur de la classe
		Param: @(epsilon): pas de l'algorithme
		@(A): (np.array) matrice des exemples à partir duquel on doit trouver b
		@b: (np.array) vecteur qui contient les résultats que l'on veut prédire
		@crit_arret: (float) tolérance de convergence
		@criterion: (str) 'grad' ou 'objective' afin 
		'''
		self.epsilon = epsilon
		self.A = A 
		self.b = b 
		self.u = np.zeros((A.shape[1],1))
		self.crit_arret = crit_arret
		self.lambd = lambd
		self.criterion = criterion

	def run_grad_prox(self,verbose=1,store_values=1):
		'''
		Permet d'effectuer le gradient proximal classique
		'''
		self.u = np.zeros((A.shape[1],1))
		crit_conv_prev = np.inf	
		crit_conv = np.inf		
		nb_iteration = 0 
		if store_values==1:
			stored_values = []
		while(crit_conv> self.crit_arret):
			u_prev = np.array(self.u) #copy
			point = np.dot(self.A.T,np.dot(self.A,self.u)-self.b)
			self.u =  thresolding_operator_vector(self.u-self.epsilon*point,self.epsilon,self.lambd)
			norm_2 = 0.5*(np.linalg.norm(np.dot(self.A,self.u)-self.b))**2
			norm_1 = np.linalg.norm(self.u,ord=1)
			if self.criterion == 'objective':	
				crit_conv = abs(crit_conv_prev-(norm_2+self.lambd*norm_1))
				crit_conv_prev = norm_2+self.lambd*norm_1
			else:
				crit_conv = (np.linalg.norm(u_prev - self.u))
			if (verbose==1):
				print('Iter numero ',nb_iteration,'0.5*||Au-b||^2=',norm_2,',|u|_1=',norm_1, 'Objective',norm_2+self.lambd*norm_1,'Conv',crit_conv)
			if store_values==1:
				stored_values.append([norm_2,norm_1,norm_2+self.lambd*norm_1,crit_conv])
			nb_iteration = nb_iteration+1
		return self.u if store_values==0 else self.u,stored_values

	def run_Fista(self,verbose=1,store_values=1):
		'''
		Permet d'effectuer FISTA
		'''
		self.u = np.zeros((A.shape[1],1))
		crit_conv_prev = np.inf	
		crit_conv = np.inf		
		nb_iteration = 0 
		v = self.u
		if store_values==1:
			stored_values = []
		while(crit_conv> self.crit_arret):
			
			u_prev = np.array(self.u) #copy
			point = np.dot(self.A.T,np.dot(self.A,v)-self.b)
			self.u =  thresolding_operator_vector(v-self.epsilon*point,self.epsilon,self.lambd)
			norm_2 = 0.5*(np.linalg.norm(np.dot(self.A,self.u)-self.b))**2
			norm_1 = np.linalg.norm(self.u,ord=1)
			if self.criterion == 'objective':	
				crit_conv = abs(crit_conv_prev-(norm_2+self.lambd*norm_1))
				crit_conv_prev = norm_2+self.lambd*norm_1
			else:
				crit_conv = (np.linalg.norm(u_prev - self.u))
			if (verbose==1):
				print('Iter numero ',nb_iteration,'0.5*||Au-b||^2=',norm_2,',|u|_1=',norm_1, 'Objective',norm_2+self.lambd*norm_1,'Conv',crit_conv)
			if store_values==1:
				stored_values.append([norm_2,norm_1,norm_2+self.lambd*norm_1,crit_conv])
			nb_iteration = nb_iteration+1
			v= self.u + (nb_iteration-1)/(nb_iteration+2)*(self.u-u_prev)
			
		return self.u if store_values==0 else self.u,stored_values



#Exemple de problèmes résolus: plusieurs lambda différent, on regarde l'erreur entre le u optimal à retrouver et le u trouvé
lambda_to_try = [0,0.01,0.05,0.1,0.5,1,5,10,100,1000,2000,5000,7000,10000]

full_outcome = []
A,b,u,L = generate_a_problem_with_u(400,200,0,100,20,40)

for lambd in lambda_to_try:
	print('lambda=',lambd)
	g_prox = GradientProximal(1/L,A,b,lambd,1e-4)
	u_norm,stored_values_norm = g_prox.run_grad_prox(verbose=0)
	u_fista,stored_values_fista = g_prox.run_Fista(verbose=0)
	
	ecart_grad_prox = np.linalg.norm(u_norm-u)
	ecart_fista = np.linalg.norm(u_fista-u)	
	list_outcome = [lambd,ecart_grad_prox,ecart_fista]

	full_outcome.append(list_outcome)

full_outcome_array = np.array(full_outcome)
full_outcome_df = pd.DataFrame(full_outcome_array,columns=['Lambda',\
                                                           'Ecart opt Grad prox','Ecart opt FISTA'])





