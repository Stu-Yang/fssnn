class Person:
    def __init__(self, newPersionName):
        self.name = newPersionName
#此处正确的，通过访问self.name的形式，实现了：
# 1.给实例中，增加了name变量
# 2.并且给name赋了初值，为newPersionName

    def sayYourName(self):
        print('My name is %s'%(self.name))
#此处由于开始正确的初始化了self对象，使得其中有了name变量，
#所以此处可以正确访问了name值了


p = Person('Bob')
p.sayYourName()   #第一种调用方法
print("-" * 50)
Person('Bob').sayYourName()#第二种调用方法