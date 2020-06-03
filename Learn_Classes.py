import turtle

class Polygon:
    def __init__(self, sides, name, size=100, color="black", line_thickness=2):
        self.sides = sides
        self.name = name
        self.size = size
        self.color = color
        self.line_thickness = line_thickness
        self.interior_angles = (self.sides-2)*180
        self.angle = self.interior_angles/self.sides

    def draw(self):
        turtle.color(self.color)
        turtle.pensize(self.line_thickness)
        for i in range(self.sides):
            turtle.forward(self.size)
            turtle.right(180-self.angle)
##        turtle.done()

##square = Polygon(4, "Square")
##pentagon = Polygon(5, "Pentagon")
##
##print(square.sides)
##print(square.name)
##print(square.interior_angles)
##print(square.angle)
##
##print(pentagon.sides)
##print(pentagon.name)
##
##hexagon = Polygon(6,"Hexagon",color="Red")
##hexagon.draw()

'''Creating Subclasses'''
class Square(Polygon):
    def __init__(self,size=100, color="black", line_thickness=2):
        super().__init__(4, "Square", size, color, line_thickness)
    ##We can choose to add to fill in the square
    def draw(self):
        turtle.begin_fill()
        super().draw()
        turtle.end_fill()

square = Square(color="#123abc",size = 200)
print(square.sides)
print(square.angle)
print(square.draw())

turtle.done()

'''Point class'''
import matplotlib.pyplot as plt
class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __add__(self, other):
        if isinstance(other, Point):
            x = self.x + other.x
            y = self.y + other.y
            return Point(x,y)
        else:
            x = self.x + other
            y = self.y + other
            return Point(x,y)

    def plot(self):
        plt.scatter(self.x, self.y)

##a = Point(1,1)
##b = Point(2,2)
##c = a + b
##a.plot()
##print(c.x, c.y)
##plt.show()

d = Point(0,2)
e = d + 5
print(e.x,e.y)
