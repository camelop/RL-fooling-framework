import struct

def parse_image_raw(r):
    magic_num, size, row, col = struct.unpack('>IIII', r[:16])
    pixels = [struct.unpack('>'+str(row*col)+'B', r[16+i*row*col:16+(1+i)*row*col]) for i in range(size)]
    return pixels, row, col

def parse_label_raw(r):
    magic_num, size= struct.unpack('>II', r[:8])
    labels = struct.unpack('>'+str(size)+'B', r[8:])
    return labels

class MnistDataset(object):


    def __init__(self, train_image_loc, train_label_loc, test_image_loc, test_label_loc):

        def load_file(loc, binary=True):
            with open(loc, 'rb' if binary else 'r') as f:
                content = f.read()
            return content
        
        self.train_image_raw = load_file(train_image_loc)
        self.train_label_raw = load_file(train_label_loc)
        self.test_image_raw = load_file(test_image_loc)
        self.test_label_raw = load_file(test_label_loc)
        
        self.preprocess()


    def preprocess(self):
        self.train_image, self.train_row, self.train_col = parse_image_raw(self.train_image_raw)
        self.train_label = parse_label_raw(self.train_label_raw)
        self.test_image, self.test_row, self.test_col = parse_image_raw(self.test_image_raw)
        self.test_label = parse_label_raw(self.test_label_raw)