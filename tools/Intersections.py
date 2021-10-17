#Inspiration for the interactive intersection selection area
#https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
#https://stackoverflow.com/questions/22140880/drawing-rectangle-or-line-using-mouse-events-in-open-cv-using-python
import cv2
import copy

def create_intersection(event,x,y,flags,params):
    intersections = params[0]
    img = copy.deepcopy(params[1])
    x_ratio = img.shape[1] / params[2]
    y_ratio = img.shape[0] / params[3]

    if event == cv2.EVENT_LBUTTONDOWN:
        print('Start of Line')
        intersections[len(intersections)+1] = intersection(x,y,x_ratio,y_ratio)

    if event == cv2.EVENT_LBUTTONUP:
        print('End of Line')
        idx = len(intersections)
        intersections[idx].end_line(x,y)

        print(intersections[idx].get_line())
        draw_intersections(intersections, copy.deepcopy(img))


    if event == cv2.EVENT_RBUTTONDOWN:
        print('deleted last line')

        if len(intersections) > 0:
            del intersections[len(intersections)]

        draw_intersections(intersections, copy.deepcopy(img))


class intersection:
    def __init__(self,x0,y0,x_ratio,y_ratio):
        self.x_ratio = x_ratio
        self.y_ratio = y_ratio

        self.x0 = int(x0*self.x_ratio)
        self.y0 = int(y0*self.y_ratio)
        self.color = (0,0,255)
        self.num_traffic = 0
        self.traffic_log = {}
        self.seen_traffic = {}
        self.frame_num = 1

    def end_line(self,x,y):
        self.x1 = int(x*self.x_ratio)
        self.y1 = int(y*self.y_ratio)
        self.x_mid_1 = int(0.49*(self.x1+self.x0))
        self.y_mid_1 = int(0.49*(self.y1+self.y0))
        self.x_mid_2 = int(0.51*(self.x1+self.x0))
        self.y_mid_2 = int(0.51*(self.y1+self.y0))

    def get_line(self):
        x0 = self.x0
        y0 = self.y0
        x1 = self.x1
        y1 = self.y1
        return (x0, y0, x1, y1)

    def out(self):
        return self.num_traffic

    def check_for_crossing(self,object,frame_num):
        line_1 = [(self.x0,self.y0),(self.x1,self.y1)]
        line_2 = [object.last(),object.current()]

        intersect = intersects(line_1,line_2)

        if intersect and not object.get_id() in self.seen_traffic.keys():
            self.num_traffic += 1

            #Traffic Log of cars seen by this intersection
            self.seen_traffic[object.get_id()] = True

            #Time of traffic logged by intersection
            if frame_num in self.traffic_log:
                self.traffic_log[frame_num].append(object.get_id())
            else:
                self.traffic_log[frame_num] = [object.get_id()]
            self.frame = frame_num

    def draw(self,img,key):
        cv2.line(img, (self.x0,self.y0), (self.x1,self.y1), self.color, 5)
        cv2.putText(img,"Intersection #" + str(key),(self.x_mid_1,self.y_mid_1),0, 0.8, (255,255,255),2)
        cv2.putText(img,"Cars Counted:" + str(self.num_traffic),(self.x_mid_2,self.y_mid_2),0, 0.8, (255,255,255),2)

def draw_intersections(intersections,img):
    for key in intersections:
        intersections[key].draw(img,key)
    img_small = cv2.resize(img, (1280, 720))
    cv2.imshow('output', img_small)

#Use code from here https://www.kite.com/python/answers/how-to-check-if-two-line-segments-intersect-in-python to check
#for the intersection of lines
def on_segment(p, q, r):
    if r[0] <= max(p[0], q[0]) and r[0] >= min(p[0], q[0]) and r[1] <= max(p[1], q[1]) and r[1] >= min(p[1], q[1]):
        return True
    return False

def orientation(p, q, r):
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if val == 0 :
        return 0
    return 1 if val > 0 else -1

def intersects(seg1, seg2):
    p1, q1 = seg1
    p2, q2 = seg2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, q1, p2) :
        return True
    if o2 == 0 and on_segment(p1, q1, q2) :
        return True
    if o3 == 0 and on_segment(p2, q2, p1) :
        return True
    if o4 == 0 and on_segment(p2, q2, q1) :
        return True
    return False