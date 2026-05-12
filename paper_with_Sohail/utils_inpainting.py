
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt


def make_holes(image, min_hole_size=50, max_hole_size=200, max_holes=10):
    """
    * Take a numpy image and split its rgb part and the mask part (also binarize the mask);
    * Randomly select a number of holes to apply to the image;
    * Build the holes_mask to be output at the end as a zero valued array of the spatial shape of the rgb_image.
    * For each hole:
        * Randomly choose its dimensions and consequentially the position of the hole in the image;
        * If the image, in the selected hole area is on average is 0.8, then select that region to be 1 valued.
    
    """
    
    rgb_image = image[:,:,:3]
    mask = image[:,:,3]
    mask = (mask > 127).astype(np.uint8) # mask is either 0 or 255, so let's make it False or True

    num_holes = np.random.randint(1, max_holes + 1)
    
    h, w, c = rgb_image.shape
    holes_mask = np.zeros((h, w), dtype=np.uint8)

    for _ in range(num_holes):
        hole_size = np.random.randint(min_hole_size, max_hole_size + 1)
        y = np.random.randint(0, h - hole_size)
        x = np.random.randint(0, w - hole_size)

        region = mask[y:y+hole_size, x:x+hole_size]
        if region.mean() > 0.8:
            holes_mask[y:y+hole_size, x:x+hole_size]=1
    
    return rgb_image, mask, holes_mask


def make_holes_with_mouse(image, max_hole_size=100):
    rgb_image = image[:, :, :3]
    mask = image[:, :, 3]
    mask = (mask > 127).astype(np.uint8)

    h, w, _ = rgb_image.shape
    holes_mask = np.zeros((h, w), dtype=np.uint8)

    fig, ax = plt.subplots()
    ax.imshow(rgb_image)
    ax.set_title(
        f"Draw holes with mouse (max {max_hole_size}px per side, press ENTER to finish)"
    )
    ax.axis("off")

    rectangles = []

    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        width = x_max - x_min
        height = y_max - y_min

        # Size constraint check
        if width > max_hole_size or height > max_hole_size:
            print(
                f"Rectangle too large "
                f"(width={width}, height={height}). "
                f"Please draw again (max {max_hole_size}px)."
            )
            ax.set_title(
                f"Rectangle too large! Max {max_hole_size}px per side. Draw again."
            )
            fig.canvas.draw_idle()
            return  # reject this rectangle

        # Accept rectangle
        rectangles.append((x_min, y_min, x_max, y_max))

        rect = plt.Rectangle(
            (x_min, y_min),
            width, height,
            edgecolor="red",
            facecolor="none",
            linewidth=2
        )
        ax.add_patch(rect)
        ax.set_title("Rectangle accepted. Draw more or press ENTER to finish.")
        fig.canvas.draw_idle()

    toggle_selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True
    )

    plt.connect(
        "key_press_event",
        lambda event: plt.close() if event.key == "enter" else None
    )
    plt.show()

    # Apply mask constraint
    for x1, y1, x2, y2 in rectangles:
        region = mask[y1:y2, x1:x2]
        if region.mean() > 0.8:
            holes_mask[y1:y2, x1:x2] = 1

    return rgb_image, mask, holes_mask


def normalize(rgb):
    rgb = rgb.astype(np.float32)
    if rgb.max() > 1.0:
        rgb /= 255.0
    return rgb


class get_data(Dataset):
    '''
    '''
    def __init__(self, root_dir, imgs_folder_name, max_hole_size, transform=None):
        self.root_dir = root_dir
        self.max_hole_size = max_hole_size
        self.transform = transform
        self.imgs_folder_name = imgs_folder_name
        self.imgs_filenames = os.listdir(os.path.join(self.root_dir, self.imgs_folder_name))
    
    def __len__(self):
        return len(self.imgs_filenames)

    def __getitem__(self, idx):
        img_full_path = os.path.join(self.root_dir, self.imgs_folder_name, self.imgs_filenames[idx])
        image = np.array(Image.open(img_full_path))          # H x W x 4
        rgb, text_mask, holes_mask = make_holes(image, max_hole_size=self.max_hole_size)


        # Normalize FIRST
        rgb = normalize(rgb)
        rgb_corrupt = rgb.copy()

        # Fill holes with true "no data" noise
        n = np.sum(holes_mask)
        rgb_corrupt[holes_mask == 1] = np.random.uniform(
            low=0.0, high=1.0, size=(n, 3)
        )

        # Stack inputs
        input_tensor = np.concatenate(
            [
                rgb_corrupt,
                text_mask[..., None].astype(np.float32),
                holes_mask[..., None].astype(np.float32)
            ],
            axis=2
        )


        return (
            torch.tensor(input_tensor).permute(2, 0, 1),
            torch.tensor(rgb).permute(2, 0, 1))