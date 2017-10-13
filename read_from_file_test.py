import reading_from_file as rf

rf.read_from_file()

x, y =  rf.get_next_element(2)

print (x)
print (y)

x, y = rf.get_next_element(2)

print (x)
print (y)

x, y = rf.get_next_element(10)

print (x)
print (y)

x2 = rf.get_test_input()
y2 = rf.get_test_output()

x = rf.get_full_training_data_set()

print(len(x[0]))
print (len(x2))
