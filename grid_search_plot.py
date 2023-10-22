import pickle
import matplotlib.pyplot as plt

etas = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
alphas = [1e-1, 5e-2, 1e-2, 5e-3]
optim_names = [(eta,alpha) for eta in etas for alpha in alphas]

color_eta = {
	1e-2:'tab:red',
	5e-3:'tab:blue',
	1e-3:'tab:green',
	5e-4:'tab:orange',
	1e-4:'tab:pink'
}

style_alpha = {
	5e-3: 'solid', 
	1e-2: 'dashed',
	5e-2: 'dotted',
	1e-1: 'dashdot'
}

# style_eta = {
# 	0.1:'solid',
# 	1e-3:'dotted',
# 	1e-7:'dashed',
# 	1e-10:'dashdot'
# }

# color_alpha = {
# 	5e-3: 'tab:red', 
# 	1e-3: 'tab:green', 
# 	5e-4: 'tab:blue', 
# 	1e-4: 'tab:orange', 
# 	5e-5: 'tab:pink'
# }

with open('losses_from_grid_search2', 'rb') as file:
	losses = pickle.load(file)

plot_lines={}

for eta in etas:
	for alpha in alphas:
		plot_lines[(eta,alpha)], = plt.semilogy([1,2,3,4,5,6], losses[(eta,alpha)], color=color_eta[eta], linestyle=style_alpha[alpha])

legend1 = plt.legend([plot_lines[(eta,5e-3)] for eta in etas], ['η='+str(eta) for eta in etas], loc=1)
plt.legend([plot_lines[(1e-2,alpha)] for alpha in alphas], ['α='+str(alpha) for alpha in alphas], loc=3)
plt.gca().add_artist(legend1)
plt.xlabel("Number of epochs")
plt.ylabel("Loss")

plt.savefig('grid_search_plot')