import numpy as np
import scipy
from scipy.stats import multivariate_normal
import scipy.stats
import helper_functions as helper

def particle_filter(observations,initial_params=None,transition_functions=None,n_particles = 10):


    observations     = np.array(observations)
    observations     = observations.reshape(len(observations),-1) 
    predicted_states = np.zeros_like(observations) 
    ndims            = observations.shape[1]

    if not initial_params:
        init_mean = np.zeros(observations.shape[1])
        init_cov  = np.eye(observations.shape[1])
    elif len(initial_params)!=2:
        raise ValueError('need 2 params')
    else:
        init_mean = initial_params[0]
        init_cov  = initial_params[1]
	
    if not transition_functions:
        pass
    else:
        obs_func      = transition_functions[0]
        trans_func    = transition_functions[1]
        obs_cov       = transition_functions[2]
        trans_cov     = transition_functions[3]		
		
    for t in range(len(observations)):
        obs_t = observations[t]	
        
		#initialize particles 
        if t == 0:
            particle_set = np.zeros((n_particles,ndims))	
			
            for p in range(n_particles):
			
                particle_set[p] = np.random.multivariate_normal(init_mean,init_cov) 

            predicted_states[t] = particle_set.mean(axis=0) 
			
        else:
            particle_set, predicted_states[t] = generate_new_particles(particle_set,obs_t,trans_func
			                                                                      ,obs_func,trans_cov,obs_cov)		

    return predicted_states			
	
def generate_new_particles(old_particles,obs_t,trans_func,obsv_func,trans_cov,obs_cov):

	n_particles   = len(old_particles)
	
	temp          = np.zeros((old_particles.shape[0],old_particles.shape[1]+1)) #stores samples and the weights
	dimension     = int(old_particles.shape[1])

	for p in range(n_particles):
		#pass particle through transition model to find the mean value
		trans_mean                 = trans_func(old_particles[p])
		assert(trans_mean.reshape(len(trans_mean),-1).T.shape[1] == dimension or trans_mean.reshape(len(trans_mean),-1).shape[1] == dimension ),'particle dimension not correct'

		
		if np.prod(trans_mean.shape) == max(trans_mean.shape):
			trans_mean = trans_mean.flatten()
		# new particle is generated from Gaussian distribution with mean = transition mean and covariance=trans_cov	
		new_state                  = np.random.multivariate_normal(trans_mean,trans_cov)
		temp[p,0:-1]  = new_state

		#find the weight of the particle. i.e. evaluate p(y_t|x_t). In this case p(y_t|x_t) is Gaussian with mean = obs_mean. 
		#the new particle is passed through the state transition model
		obs_mean      = obsv_func(new_state)
		if np.prod(obs_mean.shape) == max(obs_mean.shape):
			obs_mean = obs_mean.flatten()		
		temp[p,-1]    = multivariate_normal.pdf(obs_t, mean=obs_mean, cov=obs_cov)


	# normalize weights and resample
	temp[:,-1]    = temp[:,-1]/np.sum(temp[:,-1]) 
	idx           = np.random.choice(range(n_particles), size= n_particles, p=temp[:,-1]) #sample according to the weights	
	new_particles = temp[idx,0:-1]
	prediction    = np.mean(new_particles,axis = 0)	

	return new_particles, prediction

	
###################################### generic pf
def generic_particle_filter(observations,initial_params=None,transition_functions=None,n_particles = 10):

    #a particle filter for a generic state transition and observation model
    #the transition and observation functions must be provided as inputs.
    observations     = np.array(observations)
    observations     = observations.reshape(len(observations),-1) 
    predicted_states = np.zeros_like(observations) 
    ndims            = observations.shape[1]

    if not initial_params:
        init_mean = np.zeros(observations.shape[1])
        init_cov  = np.eye(observations.shape[1])
    elif len(initial_params)!=2:
        raise ValueError('need 2 params')
    else:
        init_mean = initial_params[0]
        init_cov  = initial_params[1]
	
    if not transition_functions:
        pass
    else:
        obs_func      = transition_functions[0]
        trans_func    = transition_functions[1]

		
    for t in range(len(observations)):
        obs_t = observations[t]	
        
		#initialize particles 
        if t == 0:
            particle_set = np.zeros((n_particles,ndims))	
			
            for p in range(n_particles):
			
                particle_set[p] = np.random.multivariate_normal(init_mean,init_cov) 

            predicted_states[t] = particle_set.mean(axis=0) 
			
        else:
            particle_set, predicted_states[t] = generate_new_particles_generic(particle_set,obs_t,trans_func
			                                                                      ,obs_func)		

    return predicted_states	



def generate_new_particles_generic(old_particles,obs_t,trans_func,obsv_func):

	n_particles   = len(old_particles)
	
	temp          = np.zeros((old_particles.shape[0],old_particles.shape[1]+1)) #stores samples and the weights
	#new_particles = np.zeros_like(old_particles)
	dimension     = int(old_particles.shape[1])

	for p in range(n_particles):
		#generate a new particle, x_t, from state transition model p(x_t|x_{t-1}). Uses the transition function
		new_state                = trans_func(old_particles[p]).flatten()
		assert new_state.shape == temp[p,0:-1].shape,'old particle and new particle shape mismatch'
		temp[p,0:-1]  = new_state

		#find the weight of the particle. i.e. evaluate p(y_t|x_t)

		temp[p,-1]    = obsv_func(obs_t,new_state) #multivariate_normal.pdf(obs_t, mean=obs_mean, cov=obs_cov)
		#1/0


	# normalize weights and resample
	temp[:,-1]    = temp[:,-1]/np.sum(temp[:,-1]) 
	idx           = np.random.choice(range(n_particles), size= n_particles, p=temp[:,-1]) #sample according to the weights	
	new_particles = temp[idx,0:-1]
	prediction    = np.mean(new_particles,axis = 0)	

	return new_particles, prediction		



	


############################# TWO STEP PARTICLE FILTER	
def particle_filter_two_step(observations,initial_params=None,transition_functions=None,n_particles = 10):


    observations     = np.array(observations)
    observations     = observations.reshape(len(observations),-1) 
    predicted_states = np.zeros_like(observations) 
    ndims            = observations.shape[1]

    if not initial_params:
        init_mean = np.zeros(observations.shape[1])
        init_cov  = np.eye(observations.shape[1])
    elif len(initial_params)!=2:
        raise ValueError('need 2 params')
    else:
        init_mean = initial_params[0]
        init_cov  = initial_params[1]
	
    if not transition_functions:
        pass
    else:
        obs_func      = transition_functions[0]
        trans_func    = transition_functions[1]
        obs_cov       = transition_functions[2]
        trans_cov     = transition_functions[3]		
		
		

		

    particle_set = np.zeros((n_particles,ndims,2))	# the last dimension is 2. it holds the particles at time t-1 and t-2 respectively
	
    for p in range(n_particles):
		
        particle_set[p,:,0] = np.random.multivariate_normal(init_mean[0],init_cov) 
        particle_set[p,:,1] = np.random.multivariate_normal(init_mean[1],init_cov) 

    predicted_states[0] = particle_set.mean(axis=0)[...,0] 
    predicted_states[1] = particle_set.mean(axis=0)[...,1] 
		
				
		
    for t in range(2,len(observations)):
        obs_t = observations[t]	
        particle_set, predicted_states[t] = generate_new_particles_two_step(particle_set,obs_t,trans_func
			                                                                      ,obs_func,trans_cov,obs_cov)		

    return predicted_states			
	
def generate_new_particles_two_step(old_particles,obs_t,trans_func,obsv_func,trans_cov,obs_cov):

	n_particles   = len(old_particles)
	
	temp          = np.zeros((old_particles.shape[0],old_particles.shape[1]+1)) #stores samples and the weights
	new_particles = np.zeros_like(old_particles)
	dimension     = int(old_particles.shape[1])

	for p in range(n_particles):
		#generate a new particle, x_t, from state transition model p(x_t|x_{t-2},x_{t-1})
		last_two_steps             = old_particles[p].T.flatten() #transpose to get the last two particles then the dimensions.
		last_two_steps             = last_two_steps.reshape(-1,last_two_steps.shape[0]).T
		
		trans_mean                 = trans_func(last_two_steps)
		assert(trans_mean.reshape(len(trans_mean),-1).T.shape[1] == dimension or trans_mean.reshape(len(trans_mean),-1).shape[1] == dimension ),'particle dimension not correct'
		#state_transition_noise     = np.random.multivariate_normal(np.zeros(dimension),trans_cov)
		if np.prod(trans_mean.shape) == max(trans_mean.shape):
			trans_mean = trans_mean.flatten()
		new_state                  = np.random.multivariate_normal(trans_mean,trans_cov)#trans_mean + state_transition_noise 
		temp[p,0:-1]  = new_state

		#find the weight of the particle. i.e. evaluate p(y_t|x_t)
		obs_mean      = obsv_func(new_state)
		if np.prod(obs_mean.shape) == max(obs_mean.shape):
			obs_mean = obs_mean.flatten()		
		temp[p,-1]    = multivariate_normal.pdf(obs_t, mean=obs_mean, cov=obs_cov)



	# normalize weights and resample
	temp[:,-1]    = temp[:,-1]/np.sum(temp[:,-1]) 
	idx           = np.random.choice(range(n_particles), size= n_particles, p=temp[:,-1]) #sample according to the weights	
	new_particles[...,0] = temp[idx,0:-1]         #freshly generated particles stored at t-1
	new_particles[...,1] = old_particles[idx,:,0] #former t-1 particles stored at t-2. Note that the former t-2 particles are dropped
	prediction    = np.mean(new_particles[...,0],axis = 0)	#prediction is the mean of the recently output particles

	return new_particles, prediction	
	
	


