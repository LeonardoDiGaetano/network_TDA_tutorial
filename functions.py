# Basic data manipulation and visualisation libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Network Libraries
import networkx as nx
import meshio
import pandas as pd
import plotly.graph_objs as go
# Magic command to load watermark
#%load_ext watermark

def openatlas(path_pos):
    """Open an atlas file with its coordinates
     
    Parameters
    ----------
    path_pos: string
        Path to the file with atlas coordinates
    
    Returns
    -------
    data: list
        A list of coordinates
          
    """
    
    positions = pd.read_csv(path_pos,header=None, delim_whitespace=True)
    
    data = [list(row.values) for _, row in positions.iterrows()]
  
    return data


def read_obj_ignore_normals(path):
    # Manually read the file to exclude vertex normals
    with open(path, 'r') as f:
        lines = f.readlines()
    # Filter out lines that start with 'vn' (vertex normal)
    filtered_lines = filter(lambda line: not line.startswith('vn'), lines)
    # Write the filtered content to a temporary string
    from io import StringIO
    modified_obj_data = StringIO(''.join(filtered_lines))
    # Use meshio to read from the modified data
    return meshio.obj._obj.read_buffer(modified_obj_data)


def shell_brain(brain_mesh):
    """Returns a brain gray shell from a fixed brain.obj file
    
    Parameters
    ----------
    brain_mesh: meshio mesh object
    
    Returns
    -------
    mesh: plotly.graph_objs._mesh3d.Mesh3d
        
    """
    
    vertices = brain_mesh.points
    #triangles = brain_mesh.cells[0][1]
    triangles = brain_mesh.cells[0].data

    x, y, z = vertices.T
    I, J, K = triangles.T
    #showlegend=True gives the option to uncover the shell
    mesh = go.Mesh3d(x=x, y=y, z=z, color='grey', i=I, j=J, k=K, opacity=0.1,
                     hoverinfo=None,showlegend = True, name ='Brain Shell'  #colorscale=pl_mygrey, #intensity=z,
                     #flatshading=True, #showscale=False
                     )
  
    return mesh #iplot(fig)

def dictpos(areas, path_pos):
    """Creates a dictionary with 3D positions for a given atlas
    This function creates a transparent shell. This is necessary for hoverinfo,
    i.e. you can find the name of the ROI using your mouse.
     
    Parameters
    ----------
    path_pos: string
        Path to the file with atlas coordinates
    
    Returns
    -------
    trace: plotly.graph_objs._mesh3d.Mesh3d
        Plotly graphical object
    x: list
        X-axis coordinates 
    y: list
        Y-axis coordinates
    z: list
        Z-axis coordinates

    """
    
    data = openatlas(path_pos)
    x=[]
    y=[]
    z=[]
    pos3d = {}
    for i in range(0, len(data)):
        pos3d[i] = (data[i][0], data[i][1], data[i][2])
        x.append(data[i][0])
        y.append(data[i][1])
        z.append(data[i][2])

    xs = []
    ys = []
    zs = []
    for i in range(0, len(data)):
        pos3d[i] = (data[i][0], data[i][1], data[i][2])
        xs.append(1.01*data[i][0])
        ys.append(1.01*data[i][1])
        zs.append(1.01*data[i][2])
   
    trace1 = go.Mesh3d(x=xs, y=ys,z=zs, alphahull=4.2, opacity=0.0005,
                       color='gray', text=areas, hoverinfo='text')
    
    return trace1, x, y, z


def matplotlib_to_plotly(cmap, pl_entries):
    """Create matplotlib color scales for plotly 
      
    Parameters
    ----------
    cmap : colormap 
        A colormap in matplotly  - Ex: jet_cmap
    pl_entries: list
        Number of entries
    
    Returns
    -------
    pl_colorsacle: list 
        A color scale from matplotlib that is readble in ploty    
    
    """
    
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([float(k*h), 'rgb' + str((C[0], C[1], C[2]))])

    return pl_colorscale

# Creating colormaps that are compatible with plotly 
magma_cmap = matplotlib.cm.get_cmap('magma')
viridis_cmap = matplotlib.cm.get_cmap('viridis')
plasma_cmap = matplotlib.cm.get_cmap('plasma')
jet_cmap = matplotlib.cm.get_cmap('jet')
inferno_cmap = matplotlib.cm.get_cmap('inferno')
Spectral_cmap = matplotlib.cm.get_cmap('Spectral')
Dark2_cmap = matplotlib.cm.get_cmap('Dark2')


# This creates a palette with 255 points.
magma = matplotlib_to_plotly(magma_cmap, 255)
viridis = matplotlib_to_plotly(viridis_cmap, 255)
plasma = matplotlib_to_plotly(plasma_cmap, 255)
jet = matplotlib_to_plotly(jet_cmap, 255)
inferno = matplotlib_to_plotly(inferno_cmap, 255)
Spectral = matplotlib_to_plotly(Spectral_cmap, 255)



from plotly.offline import iplot
def tracenodek(colorV= None,sizeV=False,node_alpha = .5, value = True, scale_factor = 50):
    """

    """
    
    if colorV == None:
        colorV = 'cornflowerblue'
    
    if sizeV==False:
        sizeV = 10
    else:
        sizeV = list(np.multiply(sizeV,scale_factor))
    #    #scale=[[0, 'darkblue'], [1, 'red']]#, [1.0, 'rgb(0, 0, 255)']]

    trace2 = go.Scatter3d(x=x, y=y, z=z, text=areas, mode='markers', name='areas',
                          marker=dict(sizemode='diameter', symbol='circle', 
                                      showscale=False, size= sizeV,
                                      colorbar = dict(  title=dict( text='Node centrality', font=dict( size=18  )), thickness=30, x=0.1, len=0.8), 
                                                      opacity=node_alpha,
                             #color=group,
                             #colorscale='Viridis',
                             #line=dict(color='rgb(50,50,50)', width=0.5)
                             #),
                           color=colorV, colorscale=Spectral, cauto=value, cmin=0, cmax=5), 
                           hoverinfo='skip', showlegend=False #name=[i for i in areas]
                           #colorbar = dict(title = 'Life<br>Expectancy'),),
                           #color=group,
                           #colorscale='Viridis',
                           #line=dict(color='rgb(50,50,50)', width=0.5)),
                           #text=labels,
                           #hoverinfo='text'
                           #colorbar=dict(thickness=15,title='random')
                           ) 
    return trace2   


path_pos = 'AAL116_coordinates.txt'
path_areas = 'AAL116_region_names.txt'
path_brainobj = './Figures/brain.obj'
# Get names of areas
n_rois = 116 
list_areas = pd.read_csv(path_areas,header=None).values
areas = [list_areas[0:n_rois,0][i] for i in range(0,n_rois)] 

path_to_obj_file = path_brainobj
brain_mesh = read_obj_ignore_normals(path_to_obj_file)
#brain_mesh =  meshio.read(path_brainobj) # Reading a brain.obj file
brain_trace = shell_brain(brain_mesh)
trace1, _, _, _ = dictpos(areas, path_pos)



# Simplicial plot
data = openatlas(path_pos) # This creates a dictionary with the positions in 3D of the nodes 

x = []
y = []
z = []
pos3d = {}
for i in range(0, len(data)):
    pos3d[i] = (data[i][0], data[i][1], data[i][2])
    x.append(data[i][0])
    y.append(data[i][1])
    z.append(data[i][2])

    
def plotHOBrain( hlink_dict, hlink_alpha=.5,color_hlink = 'sandybrown', colorV= None,sizeV=False,node_alpha = .5,  movie=False):
    """
   
    """

    trace2 = tracenodek( colorV, sizeV, node_alpha ) #tracenodek(e,i,k)
    orders = list(hlink_dict.keys())
    coor = []
    link_width = 6
    border_width = 4

    for o in orders:
        if o == 1:
            width = link_width
        else:
            width = border_width
        temp = hlink_dict[o]

        k = len(temp[0])
        
        for i in range(0, len(temp)): #Running over all cliques
            # Create a vector with the positions for each clique
            xk = [] 
            yk = []
            zk = []
            for j in range(0, len(temp[i])):
                # including the positions of the cliques
                xk.append((pos3d[temp[i][j]][0]))
                yk.append((pos3d[temp[i][j]][1]))
                zk.append((pos3d[temp[i][j]][2]))
                
            "We have to pertubate a bit one of the nodes to make a mesh3d object" 
            if len(xk) == k:
                xk.append((pos3d[temp[i][0]][0]+0.05))
                yk.append((pos3d[temp[i][0]][1]+0.05))
                zk.append((pos3d[temp[i][0]][2]+0.05))
                # These are the poligons
                coor.append(go.Mesh3d(x=xk, y=yk, z=zk, alphahull=-1, opacity=hlink_alpha,
                                    color=color_hlink[o][i],colorscale=Spectral, hoverinfo='skip', showlegend=False))
            
            # These are the lines of the cliques
            coor.append(go.Scatter3d(x=xk, y=yk, z=zk, mode='lines', 
                                     line=dict(color='black', width=width), 
                                     opacity=0.75, hoverinfo='skip', 
                                     showlegend=False))
    
    
    camera = dict(up = dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0),
                  eye=dict(x=1.7, y=0.8, z=0.225) #z=0.125,y=0.95,x=1.7
                  )
    
    layout = go.Layout(title='Motor areas H-O hubs',
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)', 
                       scene=dict(camera=camera,
                                  xaxis=dict(nticks=5, tickfont=dict(size=25, 
                                                                     color='black')), 
                                  yaxis=dict(nticks=7, tickfont=dict(size=25, 
                                                                     color='black')),
                                  zaxis=dict(nticks=7, tickfont=dict(size=25, 
                                                                     color='black')))
                       )
                                 
    tracel = go.Scatter3d(x=[0, 40, 60,-50], y=[50,-80, -70,20], z=[-40,-25,60,75],
                          mode='text', text=['front', 'right', 'back','left'],
                          textfont=dict(family='Microsoft Sans Serif'), textposition='bottom center', 
                          hoverinfo='skip', showlegend=False )
    

    coor.append(brain_trace)
    coor.append(trace2)
    coor.append(tracel)
    coor.append(trace1)
    #coor.append(trace3)
    #data=[trace1,trace2,trace3]
    data = coor
    #print(coor)
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(autosize=False, width=1200*0.8, height=800, #before 1200*0.8, height=800*0.8, 
                      margin=dict(l=50, r=50, b=100, t=100, pad=4))

    fig.update_layout(scene=dict(xaxis=dict(ticklen=20, showticklabels=False, 
                                            zeroline=False, showbackground=False), 
                                 yaxis=dict(ticklen=20, showticklabels=False, 
                                            zeroline=False, showbackground=False),
                                 zaxis=dict(ticklen=20, showticklabels=False, 
                                            zeroline=False, showbackground=False))) #,zaxis=dict(ticklen=20,showticklabels=False, zeroline=False))
    fig.update_layout(
    font_family="Microsoft Sans Serif",
    )
    if movie==False:
        fig.write_html('output_test.html')
        return iplot(fig)
    
    if movie==True:
        return fig.write_image("temp.svg")
