from bokeh.models import LinearAxis, Range1d, Label
from bokeh.plotting import figure, output_file, save
from bokeh.models import Span ,ColorBar, LinearColorMapper, BasicTicker
from bokeh.layouts import gridplot
import  numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import matplotlib.cm as cm
import matplotlib.colors as clr

##Plots 8 AEs MSE as 4 figures in 1 bokeh html. Each plot contains both phy and virt for a given resource
def plot_in_bokeh_2Axis(out_file,predictedCa,predictedPhy):
    # output to static HTML file
    output_file(out_file + ".html")

    # create a new plot with a title and axis labels
    p0 = figure(x_axis_label='Samples', y_axis_label='Physical layer MSE', plot_width=1000, plot_height=620,
                y_range = Range1d(start =-0.001, end = predictedCa[0].scored['Loss_mae'].max()+0.01))
    p0.axis.axis_label_text_font_size = "22pt"
    p0.axis.axis_label_text_font_style = "bold"
    p0.axis.major_label_text_font_size = "15pt"
    p0.axis.major_label_text_font_style = "bold"


    p1 = figure(x_axis_label='Samples', y_axis_label='Physical layer MSE', plot_width=1000, plot_height=620,
                y_range = Range1d(start =-0.001, end = predictedCa[1].scored['Loss_mae'].max()+0.01))
    p1.axis.axis_label_text_font_size = "22pt"
    p1.axis.axis_label_text_font_style = "bold"
    p1.axis.major_label_text_font_size = "15pt"
    p1.axis.major_label_text_font_style = "bold"
    p2 = figure(x_axis_label='Samples', y_axis_label='Physical layer MSE', plot_width=1000, plot_height=620,
                y_range = Range1d(start =-0.001, end = predictedCa[2].scored['Loss_mae'].max()+0.01))
    p2.axis.axis_label_text_font_size = "22pt"
    p2.axis.axis_label_text_font_style = "bold"
    p2.axis.major_label_text_font_size = "15pt"
    p2.axis.major_label_text_font_style = "bold"
    p3 = figure(x_axis_label='Samples', y_axis_label='Physical layer MSE', plot_width=1000, plot_height=620,
                y_range = Range1d(start =-0.001, end = predictedCa[3].scored['Loss_mae'].max()+0.01))
    p3.axis.axis_label_text_font_size = "22pt"
    p3.axis.axis_label_text_font_style = "bold"
    p3.axis.major_label_text_font_size = "15pt"
    p3.axis.major_label_text_font_style = "bold"

    # add a line renderer with legend and line thickness
    p0.line(np.linspace(0, predictedCa[0].scored['Loss_mae'].shape[0] - 1,
                        predictedCa[0].scored['Loss_mae'].shape[0]),
            predictedCa[0].scored['Loss_mae'], legend = 'Physical layer MSE', line_width=2,line_color='#1875d1')
    p0.extra_y_ranges = {"yPhy0": Range1d(start=-0.001, end=predictedPhy[0].scored['Loss_mae'].max()+0.01)}
    p0.add_layout(LinearAxis(y_range_name="yPhy0",axis_label="Virtual layer MSE",axis_label_text_font_size='22pt',
                             axis_label_text_font_style='bold',major_label_text_font_size='15pt',major_label_text_font_style='bold'), 'right')
    p0.line(np.linspace(0, predictedPhy[0].scored['Loss_mae'].shape[0] - 1,
                        predictedPhy[0].scored['Loss_mae'].shape[0]),
            predictedPhy[0].scored['Loss_mae'], legend = 'Virtual layer MSE', line_width=2,y_range_name='yPhy0',line_color="#f46d43")


    p1.line(np.linspace(0, predictedCa[1].scored['Loss_mae'].shape[0] - 1,
                        predictedCa[1].scored['Loss_mae'].shape[0]),
            predictedCa[1].scored['Loss_mae'], legend_label="Physical layer MSE", line_width=2,line_color='#1875d1')
    p1.extra_y_ranges = {"yPhy1": Range1d(start=-0.001, end=predictedPhy[1].scored['Loss_mae'].max() + 0.01)}
    p1.add_layout(LinearAxis(y_range_name="yPhy1",axis_label="Virtual layer MSE",axis_label_text_font_size='22pt',
                             axis_label_text_font_style='bold',major_label_text_font_size='15pt',major_label_text_font_style='bold'), 'right')
    p1.line(np.linspace(0, predictedPhy[1].scored['Loss_mae'].shape[0] - 1,
                        predictedPhy[1].scored['Loss_mae'].shape[0]),
            predictedPhy[1].scored['Loss_mae'], legend='Virtual layer MSE', line_width=2, y_range_name='yPhy1',
            line_color="#ff9800")


    p2.line(np.linspace(0, predictedCa[2].scored['Loss_mae'].shape[0] - 1,
                        predictedCa[2].scored['Loss_mae'].shape[0]),
            predictedCa[2].scored['Loss_mae'], legend_label="Physical layer MSE", line_width=2,line_color='#1875d1')
    p2.extra_y_ranges = {"yPhy2": Range1d(start=-0.001, end=predictedPhy[2].scored['Loss_mae'].max() + 0.01)}
    p2.add_layout(LinearAxis(y_range_name="yPhy2",axis_label="Virtual layer MSE",axis_label_text_font_size='22pt',
                             axis_label_text_font_style='bold',major_label_text_font_size='15pt',major_label_text_font_style='bold'), 'right')
    p2.line(np.linspace(0, predictedPhy[2].scored['Loss_mae'].shape[0] - 1,
                        predictedPhy[2].scored['Loss_mae'].shape[0]),
            predictedPhy[2].scored['Loss_mae'], legend='Virtual layer MSE', line_width=2, y_range_name='yPhy2',
            line_color="#ff9800")


    p3.line(np.linspace(0, predictedCa[3].scored['Loss_mae'].shape[0] - 1,
                        predictedCa[3].scored['Loss_mae'].shape[0]),
            predictedCa[3].scored['Loss_mae'], legend_label="Physical layer MSE", line_width=2,line_color='#1875d1')
    p3.extra_y_ranges = {"yPhy3": Range1d(start=-0.001, end=predictedPhy[3].scored['Loss_mae'].max() + 0.01)}
    p3.add_layout(LinearAxis(y_range_name="yPhy3",axis_label="Virtual layer MSE",axis_label_text_font_size='22pt',
                             axis_label_text_font_style='bold',major_label_text_font_size='15pt',major_label_text_font_style='bold'), 'right')
    p3.line(np.linspace(0, predictedPhy[3].scored['Loss_mae'].shape[0] - 1,
                        predictedPhy[3].scored['Loss_mae'].shape[0]),
            predictedPhy[3].scored['Loss_mae'], legend='Virtual layer MSE', line_width=2, y_range_name='yPhy3',
            line_color="#ff9800")



    # Horizontal line
    hline0 = Span(location=predictedCa[0].scored['Threshold1'][0], dimension='width',line_color='#1875d1',
                  line_width=3, line_dash='dotted')
    # p0.line([], [], legend_label='Threshold Ca', line_dash='dotted', line_color="#d50000",
    #         line_width=3)  # void gliphs just to add legend entry
    labelh0 = Label(x=70, y=70, x_units='screen', text='Some Stuff', render_mode='css',
          border_line_color='black', border_line_alpha=1.0,
          background_fill_color='white', background_fill_alpha=1.0)
    p0.renderers.extend([hline0])
    hline0P = Span(location=predictedPhy[0].scored['Threshold1'][0], dimension='width', line_color='#ff9800',
                  line_width=3,line_dash='dotted',y_range_name='yPhy0')
    # p0.line([], [], legend_label='Threshold Phy', line_dash='dashed', line_color="#d50000",
    #         line_width=3)  # void gliphs just to add legend entry
    p0.renderers.extend([hline0P])


    hline1 = Span(location=predictedCa[1].scored['Threshold1'][0], dimension='width',line_color='#1875d1',
                  line_width=3,line_dash='dotted')
    #p1.line([], [], legend_label='Threshold', line_dash='dotted', line_color="#d50000", line_width=3)
    p1.renderers.extend([hline1])
    hline1P = Span(location=predictedPhy[1].scored['Threshold1'][0], dimension='width', line_color='#ff9800',
                   line_width=3,line_dash='dotted',y_range_name='yPhy1')
    #p1.line([], [], legend_label='Threshold Phy', line_dash='dashed', line_color="#d50000",
         #   line_width=3)  # void gliphs just to add legend entry
    p1.renderers.extend([hline1P])


    hline2 = Span(location=predictedCa[2].scored['Threshold1'][0], dimension='width',line_color='#1875d1',
                  line_width=3,line_dash='dotted')
    #p2.line([], [], legend_label='Threshold', line_dash='dotted', line_color="#d50000", line_width=3)
    p2.renderers.extend([hline2])
    hline2P = Span(location=predictedPhy[2].scored['Threshold1'][0], dimension='width', line_color='#ff9800',
                   line_width=3, line_dash='dotted',y_range_name='yPhy2')
    #p2.line([], [], legend_label='Threshold Phy', line_dash='dashed', line_color="#d50000",
           # line_width=3)  # void gliphs just to add legend entry
    p2.renderers.extend([hline2P])


    hline3 = Span(location=predictedCa[3].scored['Threshold1'][0], dimension='width',line_color='#1875d1',
                  line_width=3, line_dash='dotted')
   # p3.line([], [], legend_label='Threshold', line_dash='dotted', line_color="#d50000", line_width=3)
    p3.renderers.extend([hline3])
    hline3P = Span(location=predictedPhy[3].scored['Threshold1'][0], dimension='width', line_color='#ff9800',
                   line_width=3,line_dash='dotted',y_range_name='yPhy3')
   # p3.line([], [], legend_label='Threshold Phy', line_dash='dashed', line_color="#d50000",
          #  line_width=3)  # void gliphs just to add legend entry
    p3.renderers.extend([hline3P])
    # output_file("name.html")
    # show the results
    p0.legend.label_text_font_size = '20pt'
    p1.legend.label_text_font_size = '20pt'
    p2.legend.label_text_font_size = '20pt'
    p3.legend.label_text_font_size = '20pt'
    p0.legend.label_text_font_style = 'bold'
    p1.legend.label_text_font_style = 'bold'
    p2.legend.label_text_font_style = 'bold'
    p3.legend.label_text_font_style = 'bold'

    p0.legend.location = "top_left"
    p1.legend.location = "top_left"
    p2.legend.location = "top_left"
    p3.legend.location = "top_left"
    grid = gridplot([[p0, p1], [p2, p3]], plot_width=1000, plot_height=620)
    save(grid)

## Plots phyVsVirt radiography in bokeh
def plot_phy_virt_radio(scoredCa, scoredPhy, typeM,testCase):
    if scoredCa.shape[0] > scoredPhy.shape[0]:
        mse_ca_out = scoredCa.iloc[:scoredPhy.shape[0], 0]  # MSE until failedCall length
        y = mse_ca_out.values
        x = scoredPhy['Loss_mae'].values
    else:
        mse_phy_out = scoredPhy.iloc[:scoredCa.shape[0], 0]  # MSE until failedCall length
        x = mse_phy_out.values
        y = scoredCa['Loss_mae'].values
    sclr = MinMaxScaler()
    x = np.asarray(sclr.fit_transform(x.reshape(-1, 1)), dtype=np.float64).reshape(x.shape[0])
    xThresh = sclr.transform(np.asarray(scoredPhy['Threshold1'][0]).reshape(-1, 1))
    y = np.asarray(sclr.fit_transform(y.reshape(-1, 1)), dtype=np.float64).reshape(y.shape[0])
    yThresh = sclr.transform(np.asarray(scoredCa['Threshold1'][0]).reshape(-1, 1))
    xmax = x.max()
    ymax = y.max()
    X, Y = np.mgrid[0:xmax:1000j, 0:ymax:1000j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    output_file("radioVirtPhy/images/"+typeM.lower()+testCase+"RadioCaPhy.html")
    p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")], toolbar_location='below',
               x_axis_label='MSE for physical '+typeM+' group',y_axis_label='MSE for virtual '+typeM+' group',
               plot_width=1000,plot_height=620,hidpi=True)
    p.axis.axis_label_text_font_size = "22pt"
    p.axis.axis_label_text_font_style = "bold"
    p.axis.major_label_text_font_size = "15pt"
    p.axis.major_label_text_font_style = "bold"
    p.x_range.range_padding = p.y_range.range_padding = 0

    colormap = cm.get_cmap("cubehelix_r")  # choose any matplotlib colormap here
    bokehpalette = [clr.rgb2hex(m) for m in colormap(np.arange(colormap.N))]

    color_mapper_lin = LinearColorMapper(palette=bokehpalette)
    p.image(image=[Z.T], x=0, y=0, dw=xmax, dh=ymax, palette=bokehpalette,level="image")
    hline0 = Span(location=yThresh[0][0], dimension='width', line_color='red', line_width=8, line_dash='dotted')
    vline0 = Span(location=xThresh[0][0], dimension='height', line_color='red', line_width=8, line_dash='dotted')
    p.renderers.extend([hline0])
    p.renderers.extend([vline0])
    p.grid.grid_line_width = 0.5
    color_bar = ColorBar(color_mapper=color_mapper_lin, ticker=BasicTicker(),
                         label_standoff=12, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')
    save(p)

## Plots virtVsServ radiography in bokeh
def plot_virt_serv_radio(metricDf, scored_x, fName,type ):
    if scored_x.shape[0] > metricDf.shape[0]:
        mseCut = scored_x.iloc[:metricDf['FailedCall(P)'].shape[0], 0]  # MSE until failedCall length
        x = mseCut.values
        y = metricDf['FailedCall(P)'].values
    else:
        failedCut = metricDf.iloc[:scored_x.shape[0], 11]  # MSE until failedCall length
        x = scored_x['Loss_mae'].values
        y = failedCut.values
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xmax = x.max()
    ymax = y.max()
    X, Y = np.mgrid[0:xmax:1000j, 0:ymax:1000j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    output_file(fName+"RadioVirtServ.html")
    p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")], toolbar_location='above',
               x_axis_label='MSE for the virtual '+type+' group', y_axis_label='Failed Calls', plot_width=1000,
               plot_height=620)
    p.axis.axis_label_text_font_size = "22pt"
    p.axis.axis_label_text_font_style = "bold"
    p.axis.major_label_text_font_size = "15pt"
    p.axis.major_label_text_font_style = "bold"
    p.x_range.range_padding = p.y_range.range_padding = 0
    colormap = cm.get_cmap("cubehelix_r")  # choose any matplotlib colormap here
    bokehpalette = [clr.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    color_mapper = LinearColorMapper(palette=bokehpalette)
    p.image(image=[Z.T], x=0, y=0, dw=xmax, dh=ymax, palette=bokehpalette, level="image")
    p.grid.grid_line_width = 0.5
    color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), border_line_color=None, location=(0, 0))
    vline0 = Span(location=scored_x['Threshold1'][0], dimension='height', line_color='red',
                  line_width=8,
                  line_dash='dotted')
    p.renderers.extend([vline0])
    p.add_layout(color_bar, 'right')
    save(p)


