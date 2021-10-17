import cv2

class hist:
    def __init__(self,track,frame_num,color):
        self.id = track.track_id
        self.type = track.get_class
        self.current_frame = frame_num
        self.color = color

        bbox = track.to_tlbr()

        #Center of Mass History
        self.COMS = {}
        self.COMS[frame_num-1] = (int((bbox[0]+bbox[2])/2), int(bbox[3]))
        self.COMS[frame_num] = (int((bbox[0]+bbox[2])/2), int(bbox[3]))

        #Consecutive History
        self.history = {}
        self.history_length = 0
        self.history[self.history_length] = (int((bbox[0]+bbox[2])/2), int(bbox[3]))
        self.history_length += 1
        self.history[self.history_length] = (int((bbox[0]+bbox[2])/2),int(bbox[3]))

    def add(self,track,frame_num):
        self.current_frame = frame_num

        bbox = track.to_tlbr()

        #Center of Mass History
        self.COMS[frame_num] = (int((bbox[0]+bbox[2])/2), int(bbox[3]))

        if not frame_num-1 in self.COMS:
            self.COMS[frame_num-1] = (int((bbox[0]+bbox[2])/2), int(bbox[3]))

        #Add to the consecutive History
        self.history_length += 1
        self.history[self.history_length] = (int((bbox[0]+bbox[2])/2),int(bbox[3]))


    def current(self):
        return self.COMS[self.current_frame]

    def last(self):
        return self.COMS[self.current_frame-1]

    def get_id(self):
        return self.id

    def draw_last_x(self,x,img):
        if self.history_length-x <= 0:
            x_lim = self.history_length
        else:
            x_lim = x
        for point in range(self.history_length-x_lim,self.history_length-1,1):
            # Draw a line connecting the history
            cv2.line(img, self.history[point], self.history[point+1], self.color, int(20*(-self.history_length+point+x_lim)/x_lim)+3)