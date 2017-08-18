#! /usr/bin/env python
# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2017 Jake Bruce and Théotime Colin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np, cv2, sys, os, math, random, time, cPickle as pickle

#--------------------------------------------------------------------

INITIAL_THRESHOLD    = 20    # initial threshold for circle detection
THRESHOLD_INC        = 1     # amount to increase or decrease threshold with user input
DS                   = 1.0   # downsampling factor
PADDING              = 200   # padding for instruction messages
POLY_ALPHA           = 0.50  # opacity of selected polygons
CIRC_ALPHA           = 0.50  # opacity of detected circles
CELL_INFLATE         = 1.3   # inflate the size of detected cells by this much to cover edges
DETECT_DS            = 0.5   # downsampling factor for circle detection, speeds up threshold adjustment step
REDETECT_ON_FULL_IMG = False # after downsampled threshold selection stage, re-detect with full img
FULLSCREEN           = True  # open cv2 window fullscreen

#--------------------------------------------------------------------

def handle_mouse(event, x, y, _, __):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corners) < 4:
            corners.append((x,y))
        elif selecting is not None:
            selection.append((x,y))

#--------------------------------------------------------------------

def select_roi(draw_text=True):
    if draw_text:
        # draw instructions
        corners_text = ["top left", "top right", "bottom right", "bottom left"]
        cv2.rectangle(viz_img,
                (int(img.shape[1]*0.2),int(img.shape[0]*0.4)),
                (int(img.shape[1]*0.8),int(img.shape[0]*0.6)), (255,255,255), -1)
        cv2.putText(viz_img, "Please click the %s corner of frame" % corners_text[len(corners)],
                (int(img.shape[1]*0.225),int(img.shape[0]*0.5)),
                cv2.FONT_HERSHEY_COMPLEX, 2.5*DS, (0,0,0), 3)

    # draw progress
    for i in range(1,len(corners)):
        cv2.line(viz_img, corners[i-1], corners[i], (0,255,0), 5)
    for corner in corners:
        cv2.circle(viz_img, corner, 20, (0,0,255), -1)

    # update the screen and get user input
    cv2.imshow("viz", viz_img)
    key = cv2.waitKey(30) & 0xff
    if key == 27: sys.exit(0) # if ESC key, quit


#--------------------------------------------------------------------

def crop_roi():
    global rot_roi, img, bgr_img, viz_img
    rot_roi = list(cv2.minAreaRect(np.array(corners)))
    if   rot_roi[1][0] > rot_roi[1][1]: rot_roi[1] = (rot_roi[1][1], rot_roi[1][0]) # if taller than wide
    if   rot_roi[2] >  45: rot_roi[2] -= 90 # if rotated by too much
    elif rot_roi[2] < -45: rot_roi[2] += 90 # if rotated by too much
    M = cv2.getRotationMatrix2D(rot_roi[0], rot_roi[2], 1.0)
    warped_img = cv2.warpAffine(src=img, M=M, dsize=(viz_img.shape[1], viz_img.shape[0]))
    extracted_roi = cv2.getRectSubPix(warped_img,
            (int(rot_roi[1][1]), int(rot_roi[1][0])),
            (int(rot_roi[0][0]), int(rot_roi[0][1])))
    img = extracted_roi.copy()
    bgr_img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    viz_img = bgr_img.copy()

    # draw instruction text
    padded_img = np.full((viz_img.shape[0]+PADDING, viz_img.shape[1], 3), (255,255,255), dtype=np.uint8)
    padded_img[:-PADDING,:,:] = viz_img
    cv2.putText(padded_img, "Detecting circles...",
            (40, viz_img.shape[0]+160),
            cv2.FONT_HERSHEY_COMPLEX, 3.0*DS, (0,0,0), 3)
    viz_img = padded_img
    cv2.imshow("viz", viz_img)
    cv2.waitKey(1)
    viz_img = bgr_img.copy()

#--------------------------------------------------------------------

def detect_circles():
    global circles, spacing, viz_img, threshold, param1, threshold_done

    viz_img = bgr_img.copy()

    # hedging for OpenCV2 vs OpenCV3
    try:    hough_flag = cv2.HOUGH_GRADIENT
    except: hough_flag = cv2.cv.CV_HOUGH_GRADIENT

    # detect circles with Hough transform
    small_img = cv2.resize(img, dsize=None, fx=DETECT_DS, fy=DETECT_DS)
    circles = cv2.HoughCircles(small_img[:,:,0], hough_flag,
            1.2,                  # dp param
            25*DS*DETECT_DS,      # min dist between centers
            None,                 # circles (?)
            param1,               # param1
            threshold,            # param2
            int(10*DS*DETECT_DS), # min radius
            int(30*DS*DETECT_DS)) # max radius

    # convert the (x, y) coordinates and radius of the circles to integers
    if circles is not None: circles = np.round(circles[0, :]/DETECT_DS).astype("int")

    # find median separation to estimate cell size
    centers = np.array([[c[0],c[1]] for c in circles], np.float32)
    best_dists = []
    for (x, y, r) in circles:
        dists = np.linalg.norm(centers - [x,y], axis=1)
        best_dists.append(dists[dists>0].min())
    spacing = np.median(best_dists)

    # draw circles
    trans_circles(viz_img, circles, (0,200,0))

    # prompt for user input to adjust the threshold
    padded_img = np.full((viz_img.shape[0]+PADDING, viz_img.shape[1], 3), (255,255,255), dtype=np.uint8)
    padded_img[:-PADDING,:,:] = viz_img

    # draw instruction text
    cv2.putText(padded_img, "Adjust empty cell detection threshold",
            (40, viz_img.shape[0]+80),
            cv2.FONT_HERSHEY_COMPLEX, 2.0*DS, (0,0,0), 3)
    cv2.putText(padded_img, "1: dec threshold       2: inc threshold         Enter: finish         ESC: cancel",
            (40, viz_img.shape[0]+170),
            cv2.FONT_HERSHEY_COMPLEX, 2.0*DS, (0,0,0), 3)
    viz_img = padded_img

    # show the detections and get user input
    cv2.imshow("viz", viz_img)
    key = 255
    while key == 255: key = cv2.waitKey(100) & 0xff

    if   key == ord('1'):
        threshold = max(1,threshold-THRESHOLD_INC)
    elif key == ord('2'):
        threshold += THRESHOLD_INC

    # this parameter is less intuitive than the threshold, not worth adding as a tuneable option
    #elif key == ord('3'):
    #    param1 = max(1,param1-1)
    #elif key == ord('4'):
    #    param1 += 1

    elif key == ord('\r'):
        threshold_done = True

        if REDETECT_ON_FULL_IMG:
            # re-detect circles with Hough transform on full-size image
            circles = cv2.HoughCircles(img[:,:,0], hough_flag,
                    1.2,        # dp param
                    25*DS,      # min dist between centers
                    None,       # circles (?)
                    param1,     # param1
                    threshold,  # param2
                    int(10*DS), # min radius
                    int(30*DS)) # max radius

            # convert the (x, y) coordinates and radius of the circles to integers
            if circles is not None: circles = np.round(circles[0, :]).astype("int")

            # find median separation to estimate cell size
            centers = np.array([[c[0],c[1]] for c in circles], np.float32)
            best_dists = []
            for (x, y, r) in circles:
                dists = np.linalg.norm(centers - [x,y], axis=1)
                best_dists.append(dists[dists>0].min())
            spacing = np.median(best_dists)

        return
    elif key == 27:
        print "Canceled. Exiting..."
        sys.exit(0)

    if key != 255:
        # re-detecting message as text feedback
        cv2.putText(viz_img, "[Re-detecting...]",
                (int(viz_img.shape[1]*0.75), viz_img.shape[0]-120),
                cv2.FONT_HERSHEY_COMPLEX, 2.0*DS, (0,0,0), 3)
        cv2.imshow("viz", viz_img)
        cv2.waitKey(1)

    viz_img = bgr_img.copy()

#--------------------------------------------------------------------

def trans_poly(img, poly, color):
    new  = np.zeros_like(img)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.fillPoly(new,  [np.array(poly)], color)
    cv2.fillPoly(mask, [np.array(poly)], 1)
    img[mask > 0,:] = POLY_ALPHA*new[mask > 0,:] + (1-POLY_ALPHA)*img[mask > 0,:]

#--------------------------------------------------------------------

def trans_circles(img, circles, color):
    new  = np.zeros_like(img)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for (x, y, r) in circles: cv2.circle(new,  (x, y), int(spacing/2.0), (0, 200, 0), -1)
    for (x, y, r) in circles: cv2.circle(mask, (x, y), int(spacing/2.0),           1, -1)
    img[mask > 0,:] = CIRC_ALPHA*new[mask > 0,:] + (1-CIRC_ALPHA)*img[mask > 0,:]

#--------------------------------------------------------------------

img_circles_drawn = None
def select_regions():
    global viz_img, selecting, selection, img_circles_drawn

    # avoid re-drawing all our detected circles if possible
    if img_circles_drawn is None:
        viz_img = bgr_img.copy()
        trans_circles(viz_img, circles, (0,200,0))
        img_circles_drawn = viz_img.copy()
    else:
        viz_img = img_circles_drawn.copy()

    padded_img = np.full((viz_img.shape[0]+PADDING, viz_img.shape[1], 3), (255,255,255), dtype=np.uint8)
    padded_img[:-PADDING,:,:] = viz_img

    if   selecting is None:
        cv2.putText(padded_img, "1: select capped brood   2: select honey   Enter: finish   ESC: cancel",
                (40, viz_img.shape[0]+130),
                cv2.FONT_HERSHEY_COMPLEX, 2.5*DS, (0,0,0), 3)
    elif selecting == 1:
        cv2.putText(padded_img, "Capped: add points to polygon, ESC: cancel, Enter: finish",
                (40, viz_img.shape[0]+130),
                cv2.FONT_HERSHEY_COMPLEX, 3.0*DS, (0,0,0), 3)
    elif selecting == 2:
        cv2.putText(padded_img, "Honey:  add points to polygon, ESC: cancel, Enter: finish",
                (40, viz_img.shape[0]+130),
                cv2.FONT_HERSHEY_COMPLEX, 3.0*DS, (0,0,0), 3)

    viz_img = padded_img

    # draw all selected capped
    for poly in capped_polygons:
        trans_poly(viz_img, poly, (128,64,64))

    # draw all selected honey
    for poly in honey_polygons:
        trans_poly(viz_img, poly, (64,128,64))

    # draw progress
    color = (255,64,64) if selecting == 1 else (64,255,64)
    if len(selection) > 1:
        trans_poly(viz_img, selection, color)
    for i in range(1,len(selection)):
        cv2.line(viz_img, selection[i-1], selection[i], color, 5)
    for point in selection:
        cv2.circle(viz_img, point, 20, (0,0,255), -1)

    # show and wait for user input
    cv2.imshow("viz", viz_img)
    key = cv2.waitKey(30) & 0xff

    # choosing which type of selection
    if selecting is None:
        if   key == ord('1'):
            selecting = 1
            selection = []
        elif key == ord('2'):
            selecting = 2
            selection = []
        elif key == ord('\r'):
            report_results()
            sys.exit(0)
        elif key == 27:
            print "Canceled. Exiting.."
            sys.exit(0)

    # currently selecting capped region
    if selecting == 1:
        if   key == ord('\r'):
            capped_polygons.append(selection)
            selection = []
            selecting = None
        elif key == 27:
            selecting = None
            selection = []

    # currently selecting honey region
    elif selecting == 2:
        if   key == ord('\r'):
            honey_polygons.append(selection)
            selection = []
            selecting = None
        elif key == 27:
            selecting = None
            selection = []

    viz_img = bgr_img.copy()

#--------------------------------------------------------------------

def report_results():
    viz_img = bgr_img.copy()

    total_area  = img.shape[0]*img.shape[1]

    capped_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for poly in capped_polygons:
        cv2.fillPoly(capped_img, [np.array(poly)], 1)

    honey_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for poly in honey_polygons:
        cv2.fillPoly(honey_img, [np.array(poly)], 1)

    cell_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    for (x, y, r) in circles: cv2.circle(cell_mask, (x,y), int(spacing/2.0*CELL_INFLATE), 0, -1)

    capped_area = (cell_mask*capped_img).astype(np.float32).sum()
    honey_area  = (cell_mask* honey_img).astype(np.float32).sum()

    capped_frac = capped_area / float(total_area)
    honey_frac  = honey_area  / float(total_area)

    # display capped area
    capped_viz = viz_img.copy()
    capped_viz[cell_mask*capped_img <= 0,:] /= 4
    padded_img = np.full((capped_viz.shape[0]+PADDING, capped_viz.shape[1], 3), (255,255,255), dtype=np.uint8)
    padded_img[:-PADDING,:,:] = capped_viz
    cv2.putText(padded_img, "Capped region %s / %s pixels, %f of total area" % (capped_area, total_area, capped_frac),
            (40, viz_img.shape[0]+160),
            cv2.FONT_HERSHEY_COMPLEX, 2.0*DS, (0,0,0), 3)
    cv2.imshow("viz", padded_img)
    cv2.waitKey(0)

    # display honey area
    honey_viz = viz_img.copy()
    honey_viz[ cell_mask*honey_img  <= 0,:] /= 4
    padded_img = np.full((honey_viz.shape[0]+PADDING, honey_viz.shape[1], 3), (255,255,255), dtype=np.uint8)
    padded_img[:-PADDING,:,:] = honey_viz
    cv2.putText(padded_img, "Honey region %s / %s pixels, %f of total area" % (honey_area, total_area, honey_frac),
            (40, viz_img.shape[0]+160),
            cv2.FONT_HERSHEY_COMPLEX, 2.0*DS, (0,0,0), 3)
    cv2.imshow("viz", padded_img)
    cv2.waitKey(0)

    # write image files as output
    cv2.imwrite(sys.argv[1].replace(".","-capped."), cell_mask*capped_img*255)
    cv2.imwrite(sys.argv[1].replace(".","-honey." ), cell_mask*honey_img *255)

    # write raw data as output as well, in case we need it later
    stats = {"img.shape"       : img.shape,
             "capped_polygons" : capped_polygons,
             "honey_polygons"  : honey_polygons,
             "circles"         : circles,
             "spacing"         : spacing}
    pickle.dump(stats, open("".join(sys.argv[1].split(".")[:-1])+"-stats.pickle", "wb"))

    # print summary
    print("========== RESULTS =========")
    print("Filename:    %s"          % sys.argv[1])

    if len(sys.argv) > 2:
        print("Frame name: %s" % sys.argv[2])

    if len(sys.argv) > 3:
        print("Side number: %s" % sys.argv[3])

    print("----------------------------")
    print("Total  area: %9d pixels"  % total_area)
    print("Honey  area: %9d pixels"  % honey_area)
    print("Capped area: %9d pixels"  % capped_area)

#--------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "Usage: %s <image-file>" % sys.argv[0]
        sys.exit(1)

    try:
        bgr_img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
        bgr_img = cv2.resize(bgr_img, dsize=None, fx=DS, fy=DS)
        img     = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    except cv2.error as e:
        errstr = "| Error reading image '%s'. Are you sure it exists? |" % sys.argv[1]
        print "-"*len(errstr)
        print errstr
        print "-"*len(errstr)
        sys.exit(1)

    corners = []
    rot_roi         = None
    circles         = None
    param1          = 200
    threshold       = INITIAL_THRESHOLD
    threshold_done  = False
    spacing         = None
    selecting       = None
    selection       = []
    capped_polygons = []
    honey_polygons  = []

    cv2.namedWindow("viz", cv2.WINDOW_NORMAL)
    if FULLSCREEN: cv2.setWindowProperty("viz", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("viz", handle_mouse)

    #--------------------------------------------------------------------

    while True:
        viz_img = bgr_img.copy()

        #---------------------------------
        # user input for drawing rectangle
        #---------------------------------
        if len(corners) < 4: select_roi()

        else:
            #---------------------------------
            # crop selected rectangle from img
            #---------------------------------
            if rot_roi is None: crop_roi()

            #---------------------------------
            # rectangle is drawn; find circles
            #---------------------------------
            if not threshold_done: detect_circles()

            #----------------------------------
            # threshold is selected, do regions
            #----------------------------------
            if threshold_done: select_regions()

