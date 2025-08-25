class person:
    number_of_people= [1,2,3,4,5,6]

    def __init__(self, name):
        self.name= name
        

p1= person("Tim")
p2= person("Vicky")

print(p1.number_of_people[3])
print(p1.name)