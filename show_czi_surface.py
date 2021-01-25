import os
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm, use
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import czi_tools as czt


def scatterplot_mpl(planetable,
                    S=0, T=0, Z=0, C=0,
                    msz2d=35,
                    normz=True,
                    fig1savename='zsurface2d.png',
                    fig2savename='zsurface3d.png',
                    msz3d=20):

    # extract XYZ positions
    try:
        xpos = planetable['X[micron]']
        ypos = planetable['Y[micron]']
        zpos = planetable['Z[micron]']
    except KeyError as e:
        xpos = planetable['X [micron]']
        ypos = planetable['Y [micron]']
        zpos = planetable['Z [micron]']

    # normalize z-data by substracting the minimum value
    if normz:
        zpos = zpos - zpos.min()

    # create a name for the figure
    figtitle = 'XYZ-Positions:  S=' + str(S) + ' T=' + str(T) + ' Z=' + str(Z) + ' CH=' + str(C)

    # try to find a "good" aspect ratio for the figures
    dx = xpos.max() - xpos.min()
    dy = ypos.max() - ypos.min()
    fsy = 8
    fsx = np.ceil(fsy * dx / dy).astype(np.int)

    # create figure
    fig1, ax1 = plt.subplots(1, 1, figsize=(fsx + 1, fsy))

    # invert the Y-axis --> O,O = Top-Left
    ax1.invert_yaxis()

    # configure the axis
    ax1.set_title(figtitle)
    ax1.set_xlabel('Stage X-Axis [micron]', fontsize=12, fontweight='normal')
    ax1.set_ylabel('Stage Y-Axis [micron]', fontsize=12, fontweight='normal')
    ax1.grid(True)
    ax1.set_aspect('equal', 'box')

    # plot data and label the colorbar
    sc1 = ax1.scatter(xpos, ypos,
                      marker='s',
                      c=zpos,
                      s=msz2d,
                      facecolor=cm.coolwarm,
                      edgecolor='black')

    # add the colorbar on the right-hand side
    cb1 = plt.colorbar(sc1,
                       fraction=0.046,
                       shrink=0.8,
                       pad=0.04)

    # add a label
    if normz:
        cb1.set_label('Z-Offset [micron]',
                      labelpad=20,
                      fontsize=12,
                      fontweight='normal')
    if not normz:
        cb1.set_label('Z-Position [micron]',
                      labelpad=20,
                      fontsize=12,
                      fontweight='normal')

    # save figure as PNG
    fig1.savefig(fig1savename, dpi=100)
    print('Saved: ', fig1savename)

    # 3D plot of surface
    fig2 = plt.figure(figsize=(fsx + 1, fsy))
    ax2 = fig2.add_subplot(111, projection='3d')

    # invert the Y-axis --> O,O = Top-Left
    ax2.invert_yaxis()

    # define the labels
    ax2.set_xlabel('Stage X-Axis [micron]',
                   fontsize=12,
                   fontweight='normal')
    ax2.set_ylabel('Stage Y-Axis [micron]',
                   fontsize=12,
                   fontweight='normal')
    ax2.set_title(figtitle)

    # plot data and label the colorbar
    sc2 = ax2.scatter(xpos, ypos, zpos,
                      marker='.',
                      s=msz3d,
                      c=zpos,
                      facecolor=cm.coolwarm,
                      depthshade=False)

    # add colorbar to the 3d plot
    cb2 = plt.colorbar(sc2, shrink=0.8)
    # add a label
    if normz:
        cb2.set_label('Z-Offset [micron]',
                      labelpad=20,
                      fontsize=12,
                      fontweight='normal')
    if not normz:
        cb2.set_label('Z-Position [micron]',
                      labelpad=20,
                      fontsize=12,
                      fontweight='normal')

    # save figure as PNG
    fig2.savefig(fig2savename, dpi=100)
    print('Saved: ', fig2savename)

    return fig1, fig2


def scatterplot_plotly(planetable,
                       S=0, T=0, Z=0, C=0,
                       msz2d=35,
                       normz=True,
                       fig1savename='zsurface2d.html',
                       fig2savename='zsurface3d.html',
                       msz3d=20):

    # extract XYZ position for the selected channel
    try:
        xpos = planetable['X[micron]']
        ypos = planetable['Y[micron]']
        zpos = planetable['Z[micron]']
    except KeyError as e:
        xpos = planetable['X [micron]']
        ypos = planetable['Y [micron]']
        zpos = planetable['Z [micron]']

    # normalize z-data by substracting the minimum value
    if normz:
        zpos = zpos - zpos.min()
        scalebar_title = 'Z-Offset [micron]'
    if not normz:
        scalebar_title = 'Z-Position [micron]'

    # create a name for the figure
    figtitle = 'XYZ-Positions:  S=' + str(S) + ' T=' + str(T) + ' Z=' + str(Z) + ' CH=' + str(C)

    fig1 = go.Figure(
        data=go.Scatter(
            x=xpos,
            y=ypos,
            mode='markers',
            text=np.round(zpos, 1),
            marker_symbol='square',
            marker_size=msz2d,
            marker=dict(
                color=zpos,
                colorscale='Viridis',
                line_width=1,
                showscale=True,
                colorbar=dict(thickness=10,
                              title=dict(
                                  text=scalebar_title,
                                  side='right'))
            )
        )
    )

    fig1.update_xaxes(showgrid=True, zeroline=True, automargin=True)
    fig1.update_yaxes(showgrid=True, zeroline=True, automargin=True)
    fig1['layout']['yaxis']['autorange'] = "reversed"
    fig1.update_layout(title=figtitle,
                       xaxis_title="StageX Position [micron]",
                       yaxis_title="StageY Position [micron]",
                       font=dict(size=16,
                                 color='Black')
                       )

    # save the figure
    fig1.write_html(fig1savename)
    print('Saved: ', fig1savename)

    fig2 = go.Figure(data=[go.Scatter3d(
        x=xpos,
        y=ypos,
        z=zpos,
        mode='markers',
        marker=dict(
            size=msz3d,
            color=zpos,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(thickness=10,
                          title=dict(
                              text=scalebar_title,
                              side='right')
                          )
        )
    )])

    fig2.update_xaxes(showgrid=True, zeroline=True, automargin=True)
    fig2.update_yaxes(showgrid=True, zeroline=True, automargin=True)
    fig2['layout']['yaxis']['autorange'] = "reversed"
    fig2.update_layout(title=figtitle,
                       xaxis_title="StageX Position [micron]",
                       yaxis_title="StageY Position [micron]",
                       font=dict(size=16,
                                 color='Black')
                       )

    # save the figure
    fig2.write_html(fig2savename)
    print('Saved: ', fig2savename)

    # show the figures for testing
    fig1.show()
    fig2.show()

    return fig1, fig2


def execute(filepath,
            separator=',',
            plot_type='html',
            msz2d=50,
            msz3d=20,
            normz=True,
            S=0, T=0, Z=0, C=0):

    print('--------------------------------------------------')
    print('FilePath : ', filepath)
    print(os.getcwd())
    print('File exists : ', os.path.exists(filepath))
    print('--------------------------------------------------')

    isczi = False
    iscsv = False

    # check if the input is a csv or czi file
    if filepath.lower().endswith('.czi'):
        isczi = True
    if filepath.lower().endswith('.csv'):
        iscsv = True

    if plot_type == 'mpl':
        saveformat = 'png'
    if plot_type == 'html':
        saveformat = 'html'

    # define name for figure to be saved
    filename = os.path.basename(filepath)
    fig1savename = os.path.splitext(filename)[0] + '_XYZ-Pos' + '.' + saveformat
    fig2savename = os.path.splitext(filename)[0] + '_XYZ-Pos3D' + '.' + saveformat

    # read the data from CSV file
    if iscsv:
        planetable = pd.read_csv(filepath, sep=separator)
    if isczi:
        # read the data from CSV file
        planetable = czt.get_czi_planetable(filepath)

    # filter the planetable for S, T, Z and C entry
    planetable_filtered = czt.filterplanetable(planetable, S=S, T=T, Z=Z, C=C)

    if plot_type == 'mpl':
        # display the XYZ positions using matplotlib
        fig1, fig2 = scatterplot_mpl(planetable_filtered,
                                     S=S, T=T, Z=Z, C=C,
                                     msz2d=msz2d,
                                     normz=normz,
                                     fig1savename=fig1savename,
                                     fig2savename=fig2savename,
                                     msz3d=msz3d)

    if plot_type == 'html':
        # display the XYZ positions using plotly
        fig1, fig2 = scatterplot_plotly(planetable_filtered,
                                        S=S, T=T, Z=Z, C=C,
                                        msz2d=msz2d,
                                        normz=normz,
                                        fig1savename=fig1savename,
                                        fig2savename=fig2savename,
                                        msz3d=msz3d)

    # write the planetable to a csv
    print('Write to CSV File : ', filename)
    csvfile = czt.save_planetable(planetable, filename,
                                  separator=separator,
                                  index=False)

    # set the outputs
    outputs = {}
    outputs['surfaceplot_2d'] = fig1savename
    outputs['surfaceplot_3d'] = fig2savename
    outputs['planetable_csv'] = csvfile

    return outputs


# Test Code locally
if __name__ == "__main__":

    #filename = r"input\Tumor_H+E_PlaneTable.csv"
    filename = r'input\testwell96_PlaneTable.csv'
    filename = r'input\test.csv'
    # filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\testwell96.czi"
    #filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Atomic\Nuclei\nuclei_RGB\H&E\Tumor_H+E.czi"
    # filename = r"L:\Data\Testdata_Zeiss\Atomic\Hocke\VoDo 20x HF.czi"
    #filename = r"D:\ImageData\Axioscan\HST20168002-TIE-BF-DAPI-nice.czi"
    #filename = r"C:\Users\m1srh\Documents\ImageSC\P2A_B6_M1_GIN-0004-Scene-2-ScanRegion1.czi"
    #filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\2 position 96 well 3 channel.czi"
    #filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\Experiment-09.czi"
    #filename = r'input\testwell96.czi'
    #filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\testwell96.czi"

    # use specific plotting backend
    use('Qt5Agg')

    execute(filename,
            separator=',',
            msz2d=70,
            msz3d=10,
            plot_type='mpl',
            normz=True,
            S=0, T=0, Z=0, C=0)

    plt.show()
