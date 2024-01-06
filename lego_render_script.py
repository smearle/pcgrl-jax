import os
import imageio


for filename in os.listdir("."):
    if filename.split(".")[-1] == "mpd":
        imgname = filename.split(".")[0] + ".png"

        leocad_command = f'''
                leocad	\
                --height "2046"	\
                --width "2046"	\
                --camera-angles 30 30	\
                --shading "full" \
                --line-width "2" \
                --aa-samples "8" \
                --image "{imgname}" \
                {filename}																	

        '''
        
        os.system(leocad_command)

imgfiles = sorted([f for f in os.listdir(".") if f.split(".")[-1] == "png"])
images = []
for im in imgfiles: 
    images.append(imageio.imread(im))


imageio.mimsave('animation.gif', images, duration=30)
