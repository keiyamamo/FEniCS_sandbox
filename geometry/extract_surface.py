import pyvista as pv
from argparse import ArgumentParser

def command_line():
    args = ArgumentParser()
    args.add_argument('--mesh', type=str, required=True)
    args.add_argument('-cp', '--camera-position', nargs=3, type=float)
    args.add_argument('-cf', '--camera-focal-point', nargs=3, type=float)
    args.add_argument('-cv','--camera-view-up', nargs=3, type=float)
    args.add_argument('--save', type=bool, default=False)

    return args.parse_args()

def main():
    args = command_line()
    mesh = pv.read(args.mesh)
    mesh.triangulate()
    
    plotter = pv.Plotter()
    plotter.subplot(0, 0)
    plotter.set_background("white")
    # Passing as a dictionary does not work for some reason
    silhouette = dict(
        color='black',
        line_width=1.5, 
        decimate=None,
        feature_angle=False,
    )
    plotter.add_mesh(mesh, 
                     color='red', 
                     silhouette=True,
                     smooth_shading=True,
                     opacity=1,
                     show_edges=False
    )
    if args.camera_position:
        plotter.camera.position = args.camera_position
        plotter.camera.focal_point = args.camera_focal_point
        plotter.camera.up = args.camera_view_up
    else:
        plotter.show()

    if args.save:
        plotter.show(cpos=plotter.camera_position, screenshot=f"{args.mesh}.png")
    
    print(f"Camera position: {plotter.camera_position[0]}")
    print(f"Camera focal point: {plotter.camera_position[1]}")
    print(f"Camera view up: {plotter.camera_position[2]}")

if __name__ == '__main__':
    main()
