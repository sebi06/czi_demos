# copied from:
# https://forum.image.sc/t/python-writing-and-appending-numpy-on-ome-tiff-files/39342/6?u=sebi06

filename = 'myimage.tif'

# define metadata for imagej
maxframe = 50
slices = 4
colors = ['GFP', 'YFP', 'RFP']
metadata = {'channels': len(colors), 'slices': slices,
            'frames': maxframe, 'hyperstack': True, 'loop': False}

for t in range(maxframe):

    # code to process images. The final output is a 3D numpy array with dimensions XYC
    # called image_stack. Here it's just a random image with increasing intensity over time
    # as a control
    image_stack = np.random.randint(0, int(t) + 1, (slices, 20, 20, 3))

    for j in range(slices):
        for i in range(len(colors)):
            imtosave = image_stack[j, :, :, i]

            # I think you need to flip to have correct orientation in imageJ
            imtosave_flip = np.flipud(imtosave.T)

            skimage.external.tifffile.imsave(
                filename,
                imtosave_flip.astype(np.uint16),
                append='force',
                imagej=True,
                metadata=metadata)
