# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFont
import sys

data = sys.argv[1]

dataOpen = open(data,'r')
info = dataOpen.readlines()
dataOpen.close()

for det in info:
		elem = det.rsplit("\t");
		for i in elem:
			print(i)
			img = Image.new('RGB', (64, 64), color = (73, 109, 137))
			fnt = ImageFont.truetype('/Library/Fonts/DevanagariMT.ttc', 50)
			d = ImageDraw.Draw(img)
			d.text((25,4), i, font=fnt, fill=(255, 255, 0))
			img.save(i+'.jpg')
		#print(elem)
