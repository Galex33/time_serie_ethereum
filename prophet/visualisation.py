import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px

def heatmap_and_cluster(df):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(8, 6))
    heat_map = sns.heatmap(corr, mask=mask, vmin=-1,cmap='RdYlBu_r', vmax=1, square=True,
                cbar_kws={"shrink": .75}, annot=True).set_title('Matrice de corrélation', fontsize=25)
    clustermap = sns.clustermap(corr, annot=True, figsize=(8,8)).fig.suptitle('Clustermap', size=25)
    
    return heat_map, clustermap
    

def scatter(df, col_x, col_y, event):
    # plot the open price
    x = df[col_x]
    y = df[col_y]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y))# Set title
    fig.update_layout(
        title_text="Time series plot of Ethereum Open Price",
    )
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            rangeslider=dict(visible=True),
            type="date",
        )
    )   
    for value in event.items(): 
        fig.add_annotation(x=value[1]['x'], y=value[1]['y'],
        text=value[1]['text'],
        showarrow=True,
        arrowhead=1)
        
    return fig

def plot(model, forecast):
    return plot_plotly(model, forecast)