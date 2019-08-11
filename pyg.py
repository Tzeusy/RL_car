import pyglet
import sys
import numpy as np 
from pyglet.gl import *

class ImageWindow(pyglet.window.Window):
    def __init__(self,*args,**kwargs):
        pyglet.window.Window.__init__(self, *args,**kwargs)
        self.update()

    def update(self, array=None):
        # the size of our texture
        if array is None:
            height, width, channels = 30, 30, 3
            data = np.random.rand(height * width, channels)
            # convert any 1's to 255
            data *= 255
        else:
            height, width, channels = array.shape
            data = array

        bytes_per_channel = 1

        # we need to flatten the array
        data = data.flatten()

        tex_data = (GLubyte * data.size)( *data.astype('uint8') )
        image = pyglet.image.ImageData(
            height,
            width,
            "RGBA",
            tex_data,
            pitch = width * channels * bytes_per_channel
        )

        self.image_sprite = pyglet.sprite.Sprite(image,
                  x=width//2, y=height//2)
        print("drawing")
        # self.image_sprite.draw()

    def on_draw(self):
        self.clear()
        self.image_sprite.draw()

# def main():
#     w = ImageWindow()

#     def foo(value):
#         w.update()

#     pyglet.clock.schedule_interval(foo, 0.01)
#     pyglet.app.run()

# main()