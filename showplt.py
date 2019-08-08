''''import matplotlib.pyplot as plt
import matplotlib.image as mpimg
pt_org = mpimg.imread('pt_org.png')
pt_alex = mpimg.imread('pt_alex.png')
pt_alfred = mpimg.imread('pt_alfred.png')
plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(pt_org)
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(pt_alex)
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(pt_alfred)
plt.axis('off')
plt.show()
'''
import svgutils.transform as sg
import sys
fig = sg.SVGFigure()
fig1 = sg.fromfile('pic\\pt_org.svg')
fig2 = sg.fromfile('pic\\pt_alex.svg')
fig3 = sg.fromfile('pic\\pt_alfred.svg')

# get the plot objects
plot1 = fig1.getroot()
plot2 = fig2.getroot()
plot3 = fig3.getroot()

#plot1.moveto(0,0, scale=0.8)
plot1.scale_xy(x=0.5,y=1)
plot2.moveto(0,400, scale=0.8)
plot3.moveto(2000,400, scale=1)

# append plots and labels to figure
fig.append([plot1,plot2,plot3])

# save generated SVG files
fig.save("pic\\fig_final.svg")

'''
from svgutils.compose import *

Figure("20cm", "20cm",
        Panel(
              SVG("pt_org.svg"),
              #Text("A", 25, 20, size=12, weight='bold')
             ),
        Panel(
              SVG("pt_alex.svg").scale(0.5),
              #Text("B", 25, 20, size=12, weight='bold')
             ).move(0, 300)
        ).save("fig_final_compose.svg")'''