import fitz # PyMuPDF
doc = fitz.open("Procedural_Generation_of_Sokoban_Levels.pdf")
for i in range(len(doc)):
    for img in doc.get_page_images(i):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha < 4:       # this is GRAY or RGB
            pix.save("p%s-%s.png" % (i, xref))
        else:               # CMYK: convert to RGB first
            pix1 = fitz.Pixmap(fitz.csRGB, pix)
            pix1.save("p%s-%s.png" % (i, xref))
            pix1 = None
        pix = None
