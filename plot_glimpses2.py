import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from utils import denormalize, bounding_box


class LoopingPillowWriter(PillowWriter):
    def finish(self):
        self._frames[0].save(
            self._outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=0)

def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--plot_dir",
        type=str,
        required=True,
        help="path to directory containing pickle dumps",
    )
    arg.add_argument("--epoch", type=int, required=True, help="epoch of desired plot")
    args = vars(arg.parse_args())
    return args["plot_dir"], args["epoch"]

# location 6 20 2   glimpse 20 32 32
def main(plot_dir, epoch):
    # read in pickle files
    glimpses = pickle.load(open(plot_dir + "g_{}.p".format(epoch), "rb"))
    locations = pickle.load(open(plot_dir + "l_{}.p".format(epoch), "rb"))

    # from ipdb import set_trace

    # set_trace()

    # glimpses = np.concatenate(glimpses)

    # grab useful params
    size = int(plot_dir.split("_")[2][0])*2
    num_anims = len(locations)
    num_cols = glimpses.shape[0]
    img_shape = glimpses.shape[1]

    # denormalize coordinates
    coords = [denormalize(img_shape, l) for l in locations]

    fig, axs = plt.subplots(nrows=1, ncols=num_cols, squeeze=False)
    # fig.set_dpi(100)

    # plot base image
    for j, ax in enumerate(axs.flat):
        ax.imshow(glimpses[j], cmap="Greys_r")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # camera = Camera(fig)
    def updateData(i):
        color = "r"
        co = coords[i]
        for j, ax in enumerate(axs.flat):
            for p in ax.patches:
                p.remove()
            c = co[j]
            rect = bounding_box(c[0], c[1], size, color)
            ax.add_patch(rect)

    # for i in range(num_anims):
    #     updateData(i)
    #     plt.pause(0.1)
    #     camera.snap()
    #
    # animation = camera.animate()
    # animation.save(plot_dir+'epoch_{}.gif'.format(epoch), writer='PillowWriter', fps=2,)
    # animate
    anim = animation.FuncAnimation(
        fig, updateData, frames=num_anims, interval=500, repeat=True
    )



    # save as gif
    writergif = LoopingPillowWriter(extra_args=["-vcodec", "h264", "-pix_fmt", "yuv420p"], fps=2)
    name = plot_dir + "epoch_{}.gif".format(epoch)
    anim.save(name, writer=writergif)

if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
