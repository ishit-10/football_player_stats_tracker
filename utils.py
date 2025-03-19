def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) // 2
    y_center = (y1 + y2) // 2
    return x_center, y_center

def get_bbox_width(bbox):
    x1, _, x2, _ = bbox
    return x2 - x1

def get_foot_position(bbox):
    x1, _, x2, y2 = bbox
    x_center = (x1 + x2) // 2
    return x_center, y2

def measure_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def measure_xy_distance(point1, point2):
    """Calculate x and y distances between two points separately"""
    x1, y1 = point1
    x2, y2 = point2
    return x2 - x1, y2 - y1 