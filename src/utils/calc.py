import math

class Calc:
    @staticmethod
    def distance(source_x, source_y, target_x, target_y):
        return math.sqrt((target_x - source_x)**2 + (target_y - source_y)**2)