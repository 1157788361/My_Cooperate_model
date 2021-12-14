class A():
    def load_data(self):
        return [1,2,3,4,5]
    def __getitem__(self, item):
        a = self.load_data()
        return a[item]
class B(A):
    def load_data(self):
        return ['a','b','c']

a = B()
print(a[1])