import numpy as np
standard = {'5':95, '4':85, '3':75, '2':65}
credit = [0.5,3,3,2,5,3,2,1,1,0.5,3,0.5,1,1,2.5,4,2,5,2,1,2,1.5,1,4,0.5,4,4,1,2,3,3,3,1.5,0.5,1.5,3,3,3,2,2,2,4,1,2,5,4,4,4,2,4,1,2,1.5,4,4,4,3,1,3,2]
grade = [5,5,3,4,4,4,5,5,5,4,5,5,5,4,5,4,4,3,4,5,5,4,4,4,5,4,5,4,4,3,3,4,3,4,3,5,5,2,3,5,4,4,5,4,3,4,3,3,4,2,4,5,5,4,4,4,4,5,5,4]
sum_ = 0
for i in range(len(credit)):
	sum_ += credit[i] * standard[str(grade[i])]
print(sum_/sum(credit))