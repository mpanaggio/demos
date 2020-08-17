import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from ipywidgets import Button, GridBox, Layout, ButtonStyle,FloatLogSlider
from ipywidgets.embed import embed_minimal_html
from ipywidgets import interactive
import seaborn as sns

def bayesian_rating_calculator(quantile=0.5,a=np.zeros(5),prior_strength=1):
    x=ss.dirichlet.rvs(a+prior_strength,size=10000)
    w=np.array(range(1,6)).reshape(-1,1)
    avg_ratings=x.dot(w)
    xq=np.quantile(avg_ratings,quantile)
    return xq,avg_ratings.squeeze()

def compare_ratings(percentile,a1,a2,prior_strength=1):
    quantile=percentile/100
    np.random.seed(2)
    xq1,x1=bayesian_rating_calculator(quantile=quantile,a=a1,prior_strength=prior_strength)
    np.random.seed(2)
    xq2,x2=bayesian_rating_calculator(quantile=quantile,a=a2,prior_strength=prior_strength)
    fig=plt.figure(figsize=(10,5))
    ax=plt.gca()
    sns.kdeplot(x1,ax=ax,shade=True,alpha=0.3,color='#1f77b4',
                label='product #1 \n 1 star: {} \n 2 star: {} \n 3 star: {} \n 4 star: {} \n 5 star: {}\n'.format(*a1))
    sns.kdeplot(x2,ax=ax,shade=True,alpha=0.3,color='#ff7f0e',
                label='product #1 \n 1 star: {} \n 2 star: {} \n 3 star: {} \n 4 star: {} \n 5 star: {}\n'.format(*a2))
    ax.axvline(xq1,color='#1f77b4',label='{:d}th percentile \n product #1 \n {:.2f} stars \n'.format(int(quantile*100),xq1),linestyle='--')
    ax.axvline(xq2,color='#ff7f0e',label='{:d}th percentile \n product #2 \n {:.2f} stars \n'.format(int(quantile*100),xq2),linestyle='--')
    ax.set_xlim(0,6)
    ax.set_xticks(range(1,6))
    ax.set_xlabel('star rating')
    ax.set_ylabel('posterior density')
    plt.legend(loc='upper left',bbox_to_anchor=(1,1.02))
    
def cr2(percentile=10,prior_strength=1,product1_1star=0,product1_2star=0,
    product1_3star=0,product1_4star=0,product1_5star=0,
    product2_1star=0,product2_2star=0,
    product2_3star=0,product2_4star=0,product2_5star=0): 
    return compare_ratings(percentile=percentile,
                    prior_strength=prior_strength,
                    a1=np.array([product1_1star,product1_2star,
                                 product1_3star,product1_4star,product1_5star]),
                    a2=np.array([product2_1star,product2_2star,
                                 product2_3star,product2_4star,product2_5star])
                                        )
interactive_plot = interactive(cr2, 
                               percentile=(1,100),
                               prior_strength=(0.01,100),
                               product1_1star=(0,100),
                               product1_2star=(0,100),
                               product1_3star=(0,100),
                               product1_4star=(0,100),
                               product1_5star=(0,100),
                               product2_1star=(0,100),
                               product2_2star=(0,100),
                               product2_3star=(0,100),
                               product2_4star=(0,100),
                               product2_5star=(0,100),
                              )
label_dict={0:'rating percentile', 1: 'prior strength',
            2: '1 star ratings (#1)',
            3: '2 star ratings (#1)',
            4: '3 star ratings (#1)',
            5: '4 star ratings (#1)',
            6: '5 star ratings (#1)',
            7: '1 star ratings (#2)',
            8: '2 star ratings (#2)',
            9: '3 star ratings (#2)',
            10: '4 star ratings (#2)',
            11: '5 star ratings (#2)'}
interactive_plot.children=[i if k !=1 else FloatLogSlider(value=1,base=10,min=-2, max=2) for k,i in enumerate(interactive_plot.children)]
style = {'description_width': 'initial'}
for k,child in enumerate(interactive_plot.children):
    if k==12:
        continue
    child.description=label_dict[k]
    child.style=style
    #help(child)
    #print(k,child)
    
order=[0,1,2,7,3,8,4,9,5,10,6,11,12]
interactive_plot.children=[interactive_plot.children[k] for k in order]
g=GridBox(children=interactive_plot.children[:-1],
        layout=Layout(
            width='100%',
            grid_template_rows='auto auto',
            grid_template_columns='45% 45%')
       )
interactive_plot.children=[g,interactive_plot.children[-1]]
#print(interactive_plot.children)
output = interactive_plot.children[-1]
output.layout.height = '500px'