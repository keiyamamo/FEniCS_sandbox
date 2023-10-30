import pyvista as pv


def plot_sol(func, min, max):
        """ Analyse 2d solution over time """
        cmap = "plasma"
        def load_time(value):
            """ Load solution at value specified using the slider """
            reader.set_active_time_value(value)
            grid = reader.read()
            grid.set_active_scalars(func)
            p.add_mesh(grid, cmap=cmap, clim=[min, max], scalar_bar_args=sargs)

        # Load in data
        xdmf_file = "xdmf_files/displacement.xdmf"
        reader = pv.get_reader(xdmf_file)
        reader.set_active_time_value(0.0)
        grid = reader.read()
        grid.set_active_scalars(func)
        # Create plot
        width = 0.6
        sargs = dict(width=width, height=0.1, position_x=(1-width)/2, position_y=0.01)
        p = pv.Plotter()
        p.add_mesh(grid, cmap=cmap, clim=[min, max], scalar_bar_args=sargs)
        p.show_bounds(xtitle="", ytitle="", ztitle="")
        p.view_xy()
        p.add_slider_widget(load_time, [0.0, 1.0], value=0.0, title="Time",
                            interaction_event="always", pointa=(0.25, 0.93), 
                            pointb=(0.75, 0.93))
        p.show()

plot_sol("Displacement", 0.0, 1e-6)