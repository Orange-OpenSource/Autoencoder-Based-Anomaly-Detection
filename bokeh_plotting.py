from bokeh.models import LinearAxis, Range1d, Label
from bokeh.plotting import figure, output_file, save
from bokeh.models import Span
from bokeh.layouts import gridplot

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
