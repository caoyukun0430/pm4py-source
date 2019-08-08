list1 = ['a', 'b', 'c', 'd']
c=[[list1[i]+list1[i+1]] for i in range(len(list1)-1)]
print(c)
